import copy
import unittest
from sparkbench.completion import analyze, counter_window, pair_analysis, quantile, valid_response


def request(index=0, mode='isolated'):
    return dict(case_id=mode+str(index), block_id=mode, selected=True, stream_kind='short', arrival_index=index,
                prompt_sha256='prompt'+str(index), planned_offset_s=index*8, output_cap=256,upstream_max_tokens=256,
                result_state='received', http_status=200, done=True, finish_reason='stop', integrity_flags=[],
                origin_input_tokens=100, origin_output_tokens=50, arrival_to_visible_s=1 if mode=='isolated' else 2,
                arrival_to_done_s=3)


class CompletionTests(unittest.TestCase):
    def test_quantile_linear_interpolation(self):
        self.assertEqual(quantile([1, 3, None], .95), 2.9)
        self.assertIsNone(quantile([], .5))

    def test_aborted_stream_not_success(self):
        row=request();row['finish_reason']='abort'
        self.assertFalse(valid_response(row))

    def test_missing_source_usage_not_filled_from_api(self):
        row=request();row['origin_output_tokens']=None;row['api_output_tokens']=50
        self.assertFalse(valid_response(row))

    def window(self):
        before=dict(requests=10,input_tokens=1000,output_tokens=500,cached_tokens=400,running=0,queued=0)
        after=dict(requests=11,input_tokens=1100,output_tokens=550,cached_tokens=450,running=0,queued=0)
        return before,after

    def test_counter_conservation(self):
        result=counter_window(*self.window(),[request()])
        self.assertTrue(result['valid']);self.assertEqual(result['cache_fraction'],.5)

    def test_counter_reset_is_not_zero_hit(self):
        before,after=self.window();after['cached_tokens']=0
        result=counter_window(before,after,[request()])
        self.assertFalse(result['valid']);self.assertIsNone(result['cache_fraction'])

    def test_unrelated_request_breaks_attribution(self):
        before,after=self.window();after['requests']=12
        self.assertFalse(counter_window(before,after,[request()])['valid'])

    def test_source_usage_must_match_counter(self):
        before,after=self.window();after['output_tokens']=551
        self.assertFalse(counter_window(before,after,[request()])['valid'])

    def test_hidden_series_reset_not_masked_by_total(self):
        before,after=self.window()
        before['series']={'requests': {'streaming':10,'nonstreaming':20}}
        after['series']={'requests': {'streaming':31,'nonstreaming':0}}
        result=counter_window(before,after,[request()])
        self.assertFalse(result['valid'])
        self.assertIn('counter_series_reset:requests',result['issues'])

    def test_nonidle_boundary_rejected(self):
        before,after=self.window();after['running']=1
        self.assertFalse(counter_window(before,after,[request()])['valid'])

    def fixture(self):
        blocks=[dict(id=m,selected=True,kind='interference',pair=0,mode=m,offered=32,planned=32,
                     local_rejections=0,source_window={'valid':True}) for m in ['isolated','mixed']]
        rows=[request(i,m) for m in ['isolated','mixed'] for i in range(32)]
        return blocks,rows

    def test_matched_pairs(self):
        p=pair_analysis(0,*self.fixture())
        self.assertTrue(p['qualified']);self.assertEqual(p['mixed_over_isolated']['median'],2)

    def test_incomplete_pair_has_no_effect_ratio(self):
        blocks,rows=self.fixture();rows[-1]['done']=False
        p=pair_analysis(0,blocks,rows)
        self.assertFalse(p['qualified']);self.assertIsNone(p['mixed_over_isolated'])

    def test_identical_counts_do_not_imply_matched_inputs(self):
        blocks,rows=self.fixture();rows[-1]['prompt_sha256']='different'
        self.assertFalse(pair_analysis(0,blocks,rows)['qualified'])

    def test_thermal_interrupt_excluded(self):
        blocks,rows=self.fixture();blocks[1]['thermal_interrupted']=True
        self.assertFalse(pair_analysis(0,blocks,rows)['qualified'])

    def test_silently_changed_output_cap_excluded(self):
        blocks,rows=self.fixture();rows[-1]['upstream_max_tokens']=128
        self.assertFalse(pair_analysis(0,blocks,rows)['qualified'])

    def test_local_rejections_not_discarded(self):
        blocks,rows=self.fixture();blocks[1]['local_rejections']=1
        self.assertFalse(pair_analysis(0,blocks,rows)['qualified'])

    def test_missing_bill_is_distinct_from_zero_charge(self):
        row={**request(), 'logical_case_id':'r0', 'attempt_started':True, 'cost_usd':None}
        data={'requests':[row], 'blocks':[], 'telemetry':[]}
        summary=analyze(data)
        self.assertFalse(summary['billing']['complete'])
        self.assertEqual(summary['billing']['missing_attempts'],[row['case_id']])
        row['cost_usd']=0.0
        summary=analyze(data)
        self.assertTrue(summary['billing']['complete'])
        self.assertEqual(summary['cost_usd_all_attempts'],0.0)

    def test_recorded_rejection_is_an_observed_outcome_not_a_complete_pair(self):
        blocks,rows=self.fixture();blocks[1]['local_rejections']=1
        rows[-1].update(result_state='local_rejection',attempt_started=False,offered_recorded=True,
                        admission_outcome='queue_expired',done=False,origin_input_tokens=None,origin_output_tokens=None)
        pair=pair_analysis(0,blocks,rows)
        self.assertTrue(pair['observed'])
        self.assertFalse(pair['qualified'])
        self.assertIsNone(pair['mixed_over_isolated'])
        self.assertEqual(pair['arms']['mixed']['short_complete'],31)

    def test_unrecorded_unsent_case_is_missing_not_observed(self):
        blocks,rows=self.fixture();blocks[1]['local_rejections']=1
        rows[-1].update(result_state='not_sent',attempt_started=False,done=False,
                        origin_input_tokens=None,origin_output_tokens=None)
        self.assertFalse(pair_analysis(0,blocks,rows)['observed'])


if __name__=='__main__':
    unittest.main()
