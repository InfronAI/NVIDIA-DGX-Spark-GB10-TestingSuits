import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock,patch
import httpx

from sparkbench.sse import Events
from sparkbench.serving import run,can_retry,validate_config

ROOT=Path(__file__).resolve().parents[1]


class Streaming(unittest.TestCase):
    def test_utf8_and_sse_boundaries_preserve_reasoning_and_content(self):
        row={'timing':{},'content':'','reasoning':'','done':False};records=[];parser=Events(row,records.append)
        first='data: '+json.dumps({'choices':[{'delta':{'reasoning_content':'想'}}]},ensure_ascii=False)+'\r\n\r\n'
        second='data: '+json.dumps({'choices':[{'delta':{'content':'你好'},'finish_reason':'stop'}],'usage':{'completion_tokens':2}},ensure_ascii=False)+'\n\n'
        wire=(first+second+'data: [DONE]\n\n').encode()
        for n,b in enumerate(wire):parser.feed(bytes([b]),n/100)
        parser.feed(b'',4,final=True)
        self.assertEqual(row['content'],'你好');self.assertEqual(row['reasoning'],'想');self.assertTrue(row['done'])
        self.assertLess(row['timing']['first_generated_s'],row['timing']['first_visible_s'])
        self.assertEqual(len(parser.times),2)

    def test_bounded_cf_retry_does_not_cover_unknown_outcomes(self):
        self.assertTrue(can_retry(502,{'cf-ray':'test'},0,2))
        self.assertFalse(can_retry(502,{},0,2))
        self.assertFalse(can_retry(None,{'cf-ray':'test'},0,2))
        self.assertFalse(can_retry(524,{'server':'cloudflare'},2,2))


class MockServing(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config=json.loads((ROOT/'configs/serving.example.json').read_text())
        self.config.update(max_requests=1,request_timeout_s=1,duration_limit_s=5)
        self.workloads=ROOT/'configs/smoke-workloads.jsonl'
        self.temp=tempfile.TemporaryDirectory();self.output=Path(self.temp.name)/'run'

    def tearDown(self):self.temp.cleanup()

    async def test_dry_run_makes_no_http_calls(self):
        def forbidden(request):raise AssertionError('dry run sent HTTP')
        result=await run(self.config,self.workloads,self.output,transport=httpx.MockTransport(forbidden))
        self.assertEqual(result['state'],'dry_run');self.assertFalse(self.output.exists())

    async def test_cf_failure_is_retained_before_success(self):
        calls=[]
        def handler(request):
            calls.append(request)
            for h in ['user-agent','accept','connection']:self.assertIn(h,request.headers)
            if len(calls)==1:return httpx.Response(502,headers={'cf-ray':'synthetic'})
            text='data: '+json.dumps({'choices':[{'delta':{'content':'ok'},'finish_reason':'stop'}],'usage':{'completion_tokens':2}})+'\n\ndata: [DONE]\n\n'
            return httpx.Response(200,text=text)
        with patch('sparkbench.serving.asyncio.sleep',new=AsyncMock()):
            result=await run(self.config,self.workloads,self.output,True,httpx.MockTransport(handler))
        self.assertEqual(result['complete'],1);self.assertEqual(len(calls),2)
        first=json.loads((self.output/'request-000001/attempt-1/result.json').read_text())
        self.assertEqual(first['state'],'http_error')
        self.assertNotIn('Authorization',json.dumps(json.loads((self.output/'summary.json').read_text())))

    async def test_transport_timeout_is_unknown_and_not_replayed(self):
        calls=[]
        def handler(request):calls.append(1);raise httpx.ReadTimeout('mock',request=request)
        result=await run(self.config,self.workloads,self.output,True,httpx.MockTransport(handler))
        self.assertEqual(result['complete'],0);self.assertEqual(len(calls),1)
        row=json.loads((self.output/'summary.json').read_text())['outcomes'][0]
        self.assertEqual(row['state'],'unknown_outcome');self.assertIsNone(result['output_tokens_s'])

    async def test_missing_done_is_not_complete(self):
        handler=lambda r:httpx.Response(200,text='data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n')
        result=await run(self.config,self.workloads,self.output,True,httpx.MockTransport(handler))
        self.assertEqual(result['complete'],0)

    async def test_malformed_sse_schema_is_recorded_without_losing_run(self):
        handler=lambda r:httpx.Response(200,text='data: {"choices":null}\n\ndata: [DONE]\n\n')
        result=await run(self.config,self.workloads,self.output,True,httpx.MockTransport(handler))
        self.assertEqual(result['complete'],0)
        row=json.loads((self.output/'summary.json').read_text())['outcomes'][0]
        self.assertEqual(row['state'],'incomplete_stream')
        self.assertTrue(row['result']['parse_errors'])

    async def test_native_completion_has_no_invented_token_timestamp(self):
        self.config.update(protocol='sglang',context_tokens=262144)
        cases=Path(self.temp.name)/'native.jsonl'
        cases.write_text(json.dumps({'id':'native','input_ids':[1,2,3]})+'\n')
        response={'text':'ok','meta_info':{'prompt_tokens':3,'completion_tokens':2,'finish_reason':{'type':'length','length':2}}}
        result=await run(self.config,cases,self.output,True,httpx.MockTransport(lambda r:httpx.Response(200,json=response)))
        self.assertEqual(result['complete'],1)
        row=json.loads((self.output/'summary.json').read_text())['outcomes'][0]['result']
        self.assertNotIn('first_generated_s',row['timing'])
        self.assertIsNone(row['timing']['average_decode_s_per_token'])

    async def test_native_abort_is_excluded_from_complete_cohort(self):
        self.config.update(protocol='sglang',context_tokens=262144)
        cases=Path(self.temp.name)/'native.jsonl'
        cases.write_text(json.dumps({'id':'native','input_ids':[1,2,3]})+'\n')
        response={'text':'partial','meta_info':{'completion_tokens':2,'finish_reason':{'type':'abort'}}}
        result=await run(self.config,cases,self.output,True,httpx.MockTransport(lambda r:httpx.Response(200,json=response)))
        self.assertEqual(result['complete'],0)
        row=json.loads((self.output/'summary.json').read_text())['outcomes'][0]
        self.assertEqual(row['state'],'incomplete_native')

    async def test_invalid_native_response_is_recorded_and_not_retried(self):
        self.config.update(protocol='sglang',context_tokens=262144)
        cases=Path(self.temp.name)/'native.jsonl'
        cases.write_text(json.dumps({'id':'native','input_ids':[1]})+'\n')
        calls=[]
        def handler(request):calls.append(1);return httpx.Response(200,json=[])
        result=await run(self.config,cases,self.output,True,httpx.MockTransport(handler))
        self.assertEqual(result['complete'],0);self.assertEqual(len(calls),1)
        row=json.loads((self.output/'summary.json').read_text())['outcomes'][0]
        self.assertEqual(row['state'],'unknown_outcome')

    async def test_existing_output_rejects_replay(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):await run(self.config,self.workloads,self.output,True,httpx.MockTransport(lambda r:httpx.Response(500)))

    def test_invalid_secrets_in_url_rejected(self):
        self.config['base_url']='https://user:pass@example.invalid/v1'
        with self.assertRaises(ValueError):validate_config(self.config)


if __name__=='__main__':unittest.main()
