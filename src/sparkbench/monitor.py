"""Timestamped Prometheus snapshots; missing telemetry remains missing."""
import asyncio
import os
import time
from .metrics import prometheus, delta


async def snapshot(client, url):
    started = time.time()
    headers={'Accept':'text/plain','User-Agent':'Infron-SparkBench/0.1','Connection':'close'}
    key=os.environ.get('SPARKBENCH_METRICS_KEY')
    if key:headers['Authorization']='Bearer '+key
    try:
        response=await client.get(url,headers=headers,timeout=10)
        response.raise_for_status()
        return {'epoch':started,'finished_epoch':time.time(),'state':'complete','text':response.text}
    except Exception as exc:
        return {'epoch':started,'finished_epoch':time.time(),'state':'missing','error_type':type(exc).__name__}


def window(before,after,names):
    if before['state']!='complete' or after['state']!='complete':
        return {'state':'missing','reason':'one or both metric snapshots missing'}
    return {'state':'observed','duration_s':after['epoch']-before['epoch'],
            'deltas':{name:delta(prometheus(before['text']),prometheus(after['text']),name) for name in names}}
