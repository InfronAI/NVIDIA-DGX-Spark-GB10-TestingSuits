"""Bounded mixed-only continuation and separation of interrupted attempts."""
import json

def schedule(root):
    p=root/'resume_authorization.json'
    if not p.exists():return [('compute',3600),('bandwidth',3600),('mixed',7200)]
    d=json.loads(p.read_text())
    if d.get('reason') not in ['user_resume_mixed_without_temperature_cutoff','user_resume_mixed_with_thermal_backoff']:return [('compute',3600),('bandwidth',3600),('mixed',7200)]
    if d.get('completed_soak_phases')!=['compute','bandwidth'] or d.get('remaining_soak_phases')!=['mixed'] or d.get('mixed_seconds')!=7200:
        raise RuntimeError('invalid mixed continuation protocol')
    if not d.get('restoration_verified'):raise RuntimeError('prior restoration not verified')
    return [('mixed',7200)]

def latest_execution(rows,node,phase):
    selected=[x for x in rows if x['node_index']==node and x.get('kind') in ['soak_baseline','sustained'] and x.get('phase')==phase]
    return max(selected,key=lambda x:x['utc']).get('execution_pid') if selected else None
