"""Portable local guards. No credentials, SSH, service stops or firmware changes."""
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

GIB=1024**3
_SAMPLES={}


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:json.dump(data,f,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def memory():
    return {k:int(v)*1024 for k,v in re.findall(r'^(\w+):\s+(\d+) kB',Path('/proc/meminfo').read_text(),re.M)}


def allowance(total,available):return max(0,min(int(available*.70),available-max(16*GIB,int(total*.15))))


def memory_check(required):
    m=memory();allowed=allowance(m['MemTotal'],m['MemAvailable'])
    return {'allowed':required<=allowed,'required_bytes':required,'allowed_bytes':allowed,'available_bytes':m['MemAvailable'],'reserve_bytes':max(16*GIB,int(m['MemTotal']*.15))}


def check_lease(root,now=None):
    root=Path(root);now=time.time() if now is None else now
    if (root/'STOP_GPU').exists():raise RuntimeError('hardware stop requested')
    session=json.loads((root/'session.json').read_text())
    if not session.get('execute') or now>=session['deadline_epoch']:raise RuntimeError('hardware session absent or expired')
    return session


def sample(root):
    root=Path(root);session=check_lease(root);now=time.time()
    prior=_SAMPLES.get(str(root))
    if prior and now-prior['epoch']<1:return prior
    process=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu',
                            '--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True,timeout=5,check=True)
    values=process.stdout.strip().splitlines()[0].split(',')
    def number(value):
        try:
            result=float(value.strip());return result if math.isfinite(result) else None
        except ValueError:return None
    gpu=dict(zip(['temperature_c','sensor_power_w','sm_clock_mhz','utilization_percent'],map(number,values)))
    if gpu['temperature_c'] is None:raise RuntimeError('GPU temperature unavailable')
    m=memory()
    if m['MemAvailable']<max(16*GIB,int(m['MemTotal']*.15)):raise RuntimeError('system memory reserve breached')
    result={'epoch':now,'gpu':gpu,'memory':m};_SAMPLES[str(root)]=result
    write_json(root/'telemetry_latest.json',result)
    with (root/'telemetry.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
    return result


def sample_guard(root):
    settings=check_lease(root);current=sample(root)
    if current['gpu']['temperature_c']<85:return current
    # A rank must never cool indefinitely inside an outstanding NCCL operation.
    # Stop the communication arm; preserve its receipts and let both ranks exit.
    if settings['mode'] in ('collectives','overlap'):
        raise RuntimeError('thermal limit reached in collective arm; abort and rerun as a new paired attempt')
    start=time.monotonic();cool_since=None
    with (Path(root)/'thermal_events.jsonl').open('a') as f:f.write(json.dumps({'utc':utc(),'state':'cooling','temperature_c':current['gpu']['temperature_c']})+'\n')
    while time.monotonic()-start<600:
        check_lease(root);time.sleep(1);current=sample(root)
        if current['gpu']['temperature_c']<75:
            if cool_since is None:cool_since=time.monotonic()
            if time.monotonic()-cool_since>=30:return current
        else:cool_since=None
    raise RuntimeError('thermal backoff exceeded 600 seconds')
