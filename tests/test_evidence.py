import csv
import json
from pathlib import Path
import tempfile
import unittest

from sparkbench.analysis import summarize
from sparkbench.registry import validate,resolve
from sparkbench.metrics import prometheus,delta,histogram,cluster_mean_ci
from sparkbench.plans import length_matrix

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/qwen3.8-flash-next-abliterated/2026-10-14_17-dual-spark'


class Evidence(unittest.TestCase):
    def test_published_coverage_does_not_claim_full_completion(self):
        value=summarize(EXP)
        self.assertTrue(value['execution_complete']);self.assertFalse(value['evidence_complete'])
        self.assertEqual(value['hardware_valid_rounds'],3740)
        self.assertEqual(value['paired_collective_trials'],840)
        self.assertEqual(value['public_length']['not_sent'],8)
        self.assertEqual(value['interference']['fully_observed_pairs'],0)
        self.assertAlmostEqual(value['capacity'][-1]['output_tokens_s'],158.89513262708914)

    def test_hardware_rows_reproduce_fixed_count(self):
        with (EXP/'data/hardware/measurements.csv').open() as handle:rows=list(csv.DictReader(handle))
        selected=[r for r in rows if r.get('state')=='measured' and r.get('kind') not in ('communication','compute_communication_overlap')]
        self.assertEqual(len(selected),3740)

    def test_registry_paths_cannot_escape(self):
        with self.assertRaises(ValueError):resolve(ROOT,'../outside')

    def test_artifact_hashes(self):self.assertEqual(validate(ROOT)['models'],1)

    def test_context_boundary_is_sum_not_individual_cap(self):
        matrix=length_matrix();self.assertEqual(len(matrix),81)
        self.assertFalse(any(r['admissible'] for r in matrix if 262144 in (r['input_tokens'],r['output_tokens'])))
        self.assertTrue(next(r for r in matrix if r['input_tokens']==r['output_tokens']==131072)['admissible'])

    def test_counter_reset_and_disappearing_labels_are_not_work(self):
        a=prometheus('x{rank="0"} 10\nx{rank="1"} 20');b=prometheus('x{rank="0"} 8\nx{rank="1"} 25')
        self.assertEqual(delta(a,b,'x')['reason'],'counter reset')
        self.assertEqual(delta(a,prometheus('x{rank="0"} 11'),'x')['reason'],'series disappeared')

    def test_histogram_reports_bounds_not_exact_quantile(self):
        a=prometheus('h_bucket{le="1"} 90\nh_bucket{le="2"} 100\nh_bucket{le="+Inf"} 100\nh_sum 60\nh_count 100')
        b=prometheus('h_bucket{le="1"} 90\nh_bucket{le="2"} 110\nh_bucket{le="+Inf"} 110\nh_sum 75\nh_count 110')
        value=histogram(a,b,'h');self.assertEqual(value['count'],10);self.assertEqual(value['mean'],1.5)
        self.assertEqual(value['quantile_bounds']['0.95'],{'lower':1.0,'upper':2.0})

    def test_single_cluster_has_no_confidence_interval(self):
        self.assertIsNone(cluster_mean_ci([{'id':'one','v':2}],lambda r:r['v'])['low'])


if __name__=='__main__':unittest.main()
