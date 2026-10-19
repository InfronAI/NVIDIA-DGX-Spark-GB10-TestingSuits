"""Cooperative cooling at independent-soak submission boundaries, never inside NCCL."""
import datetime,json,math,os,time
from pathlib import Path

DEFAULT={'enabled':True,'pause_c':85.0,'resume_below_c':75.0,'stable_seconds':30.0,'max_backoff_seconds':600.0,'poll_seconds':1.0,'scope':'independent_soak','deadline_extension':False}

def policy(root):
 p=Path(root)/'thermal_backoff_policy.json'
 if not p.exists():return {**DEFAULT,'enabled':False}
 d=json.loads(p.read_text())
 assert d.get('scope')=='independent_soak' and d.get('deadline_extension') is False
 assert 0<d['resume_below_c']<d['pause_c'] and 0<d['stable_seconds']<d['max_backoff_seconds'] and 0<d['poll_seconds']<=5
 return d

def wait_if_hot(root,check,settings,context='',clock=time.monotonic,sleep=time.sleep):
 """Return current sample and seconds spent waiting; check enforces all other guards."""
 sample=check()
 if not settings.get('enabled'):return sample,0.0
 def temperature(d):
  t=d.get('gpu',{}).get('temperature_c')
  return t if isinstance(t,(int,float)) and math.isfinite(t) else None
 temp=temperature(sample)
 if temp is not None and temp<settings['pause_c']:return sample,0.0
 start=clock();cool_since=None;root=Path(root)
 def record(state,event=True,**values):
  d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_pid':os.getpid(),'context':context,'state':state,'temperature_c':temperature(sample),'backoff_s':max(0,clock()-start),'policy':settings,**values}
  if event:
   with (root/'thermal_backoff_events.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
  tmp=root/('thermal_backoff_state.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(d));tmp.replace(root/'thermal_backoff_state.json')
 record('cooling',reason='temperature_unavailable' if temp is None else 'high_temperature')
 try:
  while True:
   # Poll lease, explicit stop, memory and telemetry age even while no CUDA work is submitted.
   sample=check();temp=temperature(sample);now=clock()
   if now-start>=settings['max_backoff_seconds']:raise RuntimeError('thermal backoff exceeded maximum duration')
   if temp is not None and temp<settings['resume_below_c']:
    if cool_since is None:cool_since=now
    if now-cool_since>=settings['stable_seconds']:
     record('resumed');return sample,now-start
   else:cool_since=None
   record('cooling',event=False)
   sleep(min(settings['poll_seconds'],settings['max_backoff_seconds']-(now-start)))
 except BaseException as exc:
  record('stopped',reason=str(exc)[:250]);raise
