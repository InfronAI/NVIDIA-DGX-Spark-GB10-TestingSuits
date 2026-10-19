"""Offline statistics and Prometheus window deltas with explicit reset handling."""
import json
import math
import re
import numpy as np

def quantile(xs,p):
    a=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(np.quantile(a,p)) if a else None

def describe(xs):
    a=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return {'n':len(a),'mean':sum(a)/len(a) if a else None,'p50':quantile(a,.5),'p90':quantile(a,.9),'p95':quantile(a,.95),'min':min(a) if a else None,'max':max(a) if a else None}

def prometheus(text):
    out={}
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        m=re.match(r'^([\w:]+)(?:\{(.*)\})?\s+([-+\w.e]+)(?:\s+\d+)?$',line)
        if not m:continue
        labels={k:json.loads('"'+v+'"') for k,v in re.findall(r'(\w+)="((?:\\.|[^"\\])*)"',m[2] or '')}
        try:value=float(m[3])
        except ValueError:continue
        if math.isfinite(value):out[(m[1],tuple(sorted(labels.items())))]=value
    return out

def selected(data,name,**labels):
    return {k:v for k,v in data.items() if k[0]==name and all(dict(k[1]).get(a)==b for a,b in labels.items())}

def value(data,name,**labels):
    xs=list(selected(data,name,**labels).values())
    return sum(xs) if xs else None

def delta(before,after,name,**labels):
    old=selected(before,name,**labels);new=selected(after,name,**labels)
    if not new:return {'value':None,'reason':'metric missing'}
    if set(old)-set(new):return {'value':None,'reason':'series disappeared'}
    ds=[v-old.get(k,0) for k,v in new.items()]
    if any(x<0 for x in ds):return {'value':None,'reason':'counter reset'}
    return {'value':sum(ds),'reason':None,'new_series':len(set(new)-set(old))}

def histogram(before,after,name):
    count=delta(before,after,name+'_count');total=delta(before,after,name+'_sum')
    groups={}
    for key in selected(after,name+'_bucket'):
        label=dict(key[1]);le=label.pop('le');group=tuple(sorted(label.items()))
        d=delta(before,after,name+'_bucket',**dict(key[1]))
        if d['value'] is None:return {'count':None,'mean':None,'reason':d['reason']}
        groups.setdefault(group,[]).append((float(le),d['value']))
    buckets={}
    for items in groups.values():
        for edge,n in items:buckets[edge]=buckets.get(edge,0)+n
    # Keep coarse quantile bounds; never invent exact histogram quantiles.
    bounds={}
    for q in [.5,.95]:
        previous=0
        for edge,n in sorted(buckets.items()):
            if count['value'] and n>=q*count['value']:
                bounds[str(q)]={'lower':previous,'upper':edge if math.isfinite(edge) else None};break
            previous=edge
    n=count['value'];s=total['value']
    return {'count':n,'mean':s/n if n and s is not None else None,'quantile_bounds':bounds,'reason':count['reason'] or total['reason']}

def cluster_mean_ci(rows,field,cluster='block',seed=20261014):
    groups={}
    for r in rows:
        v=field(r)
        if v is not None:groups.setdefault(r.get(cluster) or r['id'],[]).append(float(v))
    if len(groups)<2:return {'low':None,'high':None,'clusters':len(groups),'resamples':0,'note':'fewer than two clusters'}
    rng=np.random.default_rng(seed);sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    idx=rng.integers(0,len(groups),size=(10000,len(groups)))
    vals=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    return {'low':float(np.quantile(vals,.025)),'high':float(np.quantile(vals,.975)),'clusters':len(groups),'resamples':10000,'note':'few-block intervals unstable' if len(groups)<10 else 'cluster bootstrap; observational sample'}

def block_rate_ci(blocks,seed=20261014):
    a=[x for x in blocks if x.get('window_s',0)>0]
    if any(x.get('output_tokens') is None for x in a):return None
    if len(a)<2:return None
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(a),size=(10000,len(a)))
    tok=np.array([x['output_tokens'] for x in a]);sec=np.array([x['window_s'] for x in a]);vals=tok[idx].sum(axis=1)/sec[idx].sum(axis=1)
    return {'low':float(np.quantile(vals,.025)),'high':float(np.quantile(vals,.975)),'blocks':len(a),'resamples':10000,'note':'only 3–4 blocks: unstable interval'}
