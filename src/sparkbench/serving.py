"""Bounded OpenAI/SGLang requests with separate attempts and explicit outcomes."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from .plans import load_cases
from .sse import Events
from .monitor import snapshot, window


def validate_config(config):
    url=urlsplit(config['base_url'])
    if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('base_url must be HTTP(S) without embedded credentials/query')
    for key,low,high in [('concurrency',1,64),('max_requests',1,10000),('max_tokens',1,262144),('cf_retries',0,5)]:
        value=config[key]
        if type(value) is not int or not low<=value<=high:raise ValueError('Invalid '+key)
    if not 0<config['request_timeout_s']<=config['duration_limit_s']<=172800:
        raise ValueError('Finite request/run deadlines are required')
    if config.get('protocol','openai') not in ('openai','sglang'):
        raise ValueError('Unknown protocol')
    if config.get('arrival_rate') is not None and config['arrival_rate']<=0:
        raise ValueError('arrival_rate must be positive')
    if config.get('extra_body',{}).keys() & {'model','messages','input_ids','stream','max_tokens','sampling_params'}:
        raise ValueError('extra_body cannot replace workload identity or measurement fields')


def can_retry(status,headers,attempt,maximum):
    return (attempt<maximum and status in {502,503,504,520,521,522,523,524,525,526,527}
            and ('cf-ray' in headers or 'cloudflare' in headers.get('server','').lower()))


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def payload(config,case):
    if config.get('protocol','openai')=='sglang':
        if 'input_ids' not in case:raise ValueError('Native /generate requires tokenizer-verified input_ids')
        if len(case['input_ids'])+config['max_tokens']>config['context_tokens']:
            raise ValueError('input + output exceeds native context')
        return {'input_ids':case['input_ids'],'stream':False,'sampling_params':{
            'max_new_tokens':config['max_tokens'],'temperature':0,'ignore_eos':config.get('ignore_eos',False)}}
    if 'messages' not in case:raise ValueError('OpenAI protocol requires messages')
    return {'model':config['model'],'messages':case['messages'],'max_tokens':config['max_tokens'],
            'temperature':0,'stream':True,'stream_options':{'include_usage':True},**config.get('extra_body',{})}


async def request(client,config,case,path,deadline=None):
    headers={'User-Agent':'Infron-SparkBench/0.1','Accept':'text/event-stream' if config.get('protocol','openai')=='openai' else 'application/json','Connection':'keep-alive'}
    key=os.environ.get(config.get('api_key_env','SPARKBENCH_API_KEY'))
    if key:headers['Authorization']='Bearer '+key
    body=payload(config,case)
    native=config.get('protocol','openai')=='sglang'
    endpoint=config['base_url'].rstrip('/')+('/generate' if native else '/chat/completions')
    outcomes=[]
    for attempt in range(config['cf_retries']+1):
        remaining=deadline-time.monotonic() if deadline is not None else config['request_timeout_s']
        if remaining<=0:break
        timeout=min(config['request_timeout_s'],remaining)
        attempt_dir=path/f'attempt-{attempt+1}';attempt_dir.mkdir(parents=True)
        row={'id':case['id'],'attempt':attempt+1,'state':'started','start_epoch':time.time(),
             'done':False,'content':'','reasoning':'','timing':{},'usage_source':'endpoint_reported',
             'payload_sha256':hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()}
        dump(attempt_dir/'started.json',{k:v for k,v in row.items() if k not in ('content','reasoning')})
        started=time.perf_counter();status=None;response_headers={}
        events=[];parser=Events(row,events.append)
        try:
            async def send():
                nonlocal status,response_headers
                if native:
                    response=await client.post(endpoint,headers=headers,json=body,timeout=timeout)
                    status=response.status_code;response_headers=dict(response.headers)
                    row['status']=status
                    if status!=200:row['state']='http_error';return
                    data=response.json()
                    if not isinstance(data,dict):raise ValueError('Native response must be an object')
                    meta=data.get('meta_info') or {}
                    if not isinstance(meta,dict) or not isinstance(data.get('text',''),str):
                        raise ValueError('Invalid native response fields')
                    finish=meta.get('finish_reason')
                    finish_type=finish.get('type') if isinstance(finish,dict) else finish
                    valid=(type(meta.get('completion_tokens')) is int and meta['completion_tokens']>=0
                           and finish_type in ('stop','length') and not data.get('error'))
                    row.update(state='complete' if valid else 'incomplete_native',done=valid,content=data.get('text',''),native_meta=meta,
                        usage={'prompt_tokens':meta.get('prompt_tokens'),'completion_tokens':meta.get('completion_tokens')})
                    row['finish_reason']=meta.get('finish_reason')
                    # Native non-streaming has no observed first-token timestamp.
                    return
                async with client.stream('POST',endpoint,headers=headers,json=body,timeout=timeout) as response:
                    status=response.status_code;response_headers=dict(response.headers)
                    row['status']=status;row['timing']['headers_s']=time.perf_counter()-started
                    if status!=200:
                        row['state']='http_error';return
                    async for chunk in response.aiter_bytes():parser.feed(chunk,time.perf_counter()-started)
                    parser.feed(b'',time.perf_counter()-started,final=True)
                    row['state']='complete' if row['done'] and row.get('finish_reason') and not row.get('stream_error') and not row.get('parse_errors') else 'incomplete_stream'
            await asyncio.wait_for(send(),timeout=timeout)
        except (httpx.HTTPError,asyncio.TimeoutError,ValueError) as exc:
            row.update(state='unknown_outcome',error_type=type(exc).__name__)
        row['end_epoch']=time.time();row['timing']['e2e_s']=time.perf_counter()-started
        tokens=(row.get('usage') or {}).get('completion_tokens')
        first=row['timing'].get('first_generated_s')
        row['timing']['average_decode_s_per_token']=((row['timing']['e2e_s']-first)/(tokens-1)
            if isinstance(tokens,(int,float)) and tokens>1 and first is not None else None)
        row['output_chars']=len(row.pop('content'));row['reasoning_chars']=len(row.pop('reasoning'))
        # Store event timing and event type, not prompts, full completions or headers.
        row['event_receipt_s']=parser.times
        dump(attempt_dir/'result.json',row);outcomes.append(row)
        if not can_retry(status,response_headers,attempt,config['cf_retries']):break
        remaining=deadline-time.monotonic() if deadline is not None else float('inf')
        if remaining<=0:break
        await asyncio.sleep(min(2**attempt,16,remaining))
    if not outcomes:return {'id':case['id'],'state':'not_admitted','reason':'deadline_before_dispatch'}
    return {'id':case['id'],'state':outcomes[-1]['state'],'attempts':len(outcomes),'result':outcomes[-1]}


async def run(config,workloads,output,execute=False,transport=None):
    validate_config(config)
    cases=load_cases(workloads,config['max_requests'])
    for case in cases:payload(config,case)
    plan={'model':config['model'],'protocol':config.get('protocol','openai'),'requests':len(cases),
          'concurrency':config['concurrency'],'attempt_ceiling':len(cases)*(config['cf_retries']+1),
          'max_output_tokens_per_request':config['max_tokens'],'duration_limit_s':config['duration_limit_s']}
    if not execute:return {'state':'dry_run','plan':plan}
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    dump(root/'plan.json',plan)
    started=time.monotonic();deadline=started+config['duration_limit_s'];semaphore=asyncio.Semaphore(config['concurrency'])
    rows=[];active=0
    async with httpx.AsyncClient(transport=transport,trust_env=False,follow_redirects=False) as client:
        before=await snapshot(client,config['metrics_url']) if config.get('metrics_url') else None
        async def one(case,offered):
            nonlocal active
            if time.monotonic()>=deadline or (root/'STOP').exists():
                return {'id':case['id'],'state':'not_admitted','reason':'deadline_or_stop'}
            if config.get('arrival_rate') and semaphore.locked():
                return {'id':case['id'],'state':'local_rejection','reason':'concurrency_cap'}
            async with semaphore:
                if time.monotonic()>=deadline or (root/'STOP').exists():return {'id':case['id'],'state':'not_admitted','reason':'deadline_or_stop'}
                remain=deadline-time.monotonic()
                if remain<config['request_timeout_s']:return {'id':case['id'],'state':'not_admitted','reason':'insufficient_remaining_deadline'}
                casepath=root/f'request-{cases.index(case)+1:06d}'
                row=await request(client,config,case,casepath,deadline)
                row['offered_elapsed_s']=offered;return row
        tasks=[]
        try:
            for index,case in enumerate(cases):
                if config.get('arrival_rate'):
                    target=started+index/config['arrival_rate']
                    if target>=deadline:
                        rows.extend({'id':c['id'],'state':'not_admitted','reason':'arrival_outside_duration'} for c in cases[index:]);break
                    await asyncio.sleep(max(0,target-time.monotonic()))
                tasks.append(asyncio.create_task(one(case,time.monotonic()-started)))
            rows.extend(await asyncio.gather(*tasks))
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        after=await snapshot(client,config['metrics_url']) if before else None
        elapsed=time.monotonic()-started
    complete=[r['result'] for r in rows if r['state']=='complete']
    counted=[r for r in complete if isinstance((r.get('usage') or {}).get('completion_tokens'),(int,float))]
    summary={'state':'execution_ended','planned':len(cases),'complete':len(complete),'outcomes':rows,
             'wall_s':elapsed,'output_tokens_s':sum(r['usage']['completion_tokens'] for r in counted)/elapsed if len(counted)==len(complete) and complete else None,
             'throughput_scope':'All-run wall time, including admission waits, retries and endpoint work; endpoint-reported counts.',
             'timing_scope':'SSE event receipt times; not exact per-token or pure GPU timing.'}
    if before:
        dump(root/'metrics-before.json',before);dump(root/'metrics-after.json',after)
        summary['metrics_window']=window(before,after,config.get('counter_names',[]))
    dump(root/'summary.json',summary)
    return {k:v for k,v in summary.items() if k!='outcomes'}
