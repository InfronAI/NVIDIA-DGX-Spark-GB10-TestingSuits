"""Archived, fragmentation-safe OpenAI SSE parser; receipt times are not token clocks."""
import codecs
import json

class Events:
    """SSE data parser; event timestamps are client receipt times, not GPU token times."""
    def __init__(self, result, emit):
        self.result=result;self.emit=emit;self.decoder=codecs.getincrementaldecoder('utf-8')();self.buffer='';self.lines=[];self.times=[]

    def event(self,t):
        if not self.lines:return
        raw='\n'.join(self.lines);self.lines=[];self.emit({'elapsed_s':t,'data':raw})
        r=self.result
        if raw=='[DONE]':r['done']=True;r['timing']['done_s']=t;return
        try:j=json.loads(raw)
        except ValueError:r.setdefault('parse_errors',[]).append(raw[:500]);return
        if not isinstance(j,dict):r.setdefault('parse_errors',[]).append('non-object event');return
        for name in ['request_id','provider','model','id']:
            if j.get(name):r['response_'+name]=j[name]
        if j.get('usage') is not None:
            if not isinstance(j['usage'],dict):r.setdefault('parse_errors',[]).append('non-object usage');return
            r['usage']=j['usage']
        if j.get('error'):r['stream_error']=j['error']
        choices=j.get('choices',[])
        if not isinstance(choices,list):r.setdefault('parse_errors',[]).append('non-list choices');return
        for choice in choices:
            if not isinstance(choice,dict):r.setdefault('parse_errors',[]).append('non-object choice');continue
            if choice.get('finish_reason'):r['finish_reason']=choice['finish_reason']
            d=choice.get('delta') or {}
            if not isinstance(d,dict):r.setdefault('parse_errors',[]).append('non-object delta');continue
            content=d.get('content') or '';reason=d.get('reasoning') or d.get('reasoning_content') or ''
            tools=d.get('tool_calls') or []
            if not isinstance(content,str) or not isinstance(reason,str) or not isinstance(tools,list):
                r.setdefault('parse_errors',[]).append('invalid delta field types');continue
            if content or reason or tools:
                r['timing'].setdefault('first_generated_s',t);r['timing']['last_generated_s']=t;self.times.append(t)
            if content:r['timing'].setdefault('first_visible_s',t);r['content']+=content
            if reason:r['timing'].setdefault('first_reasoning_s',t);r['reasoning']+=reason
            if tools:r.setdefault('tool_deltas',[]).append(tools)

    def feed(self,data,t,final=False):
        self.buffer+=self.decoder.decode(data,final=final)
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1);line=line.rstrip('\r')
            if line.startswith('data:'):self.lines.append(line[5:].removeprefix(' '))
            elif line=='':self.event(t)
        if final:
            if self.buffer.startswith('data:'):self.lines.append(self.buffer[5:].removeprefix(' ').rstrip('\r'))
            self.buffer='';self.event(t)
