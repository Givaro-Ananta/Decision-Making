import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('diagnosis',Path(__file__).resolve().parents[1]/'scripts/diagnose_markov_050_validation.py')
diagnosis=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnosis)
class DiagnosisTests(unittest.TestCase):
    def test_rank_uses_true_destination_and_shared_ties(self):
        p=np.array([[.7,.2,.08,.02],[.4,.4,.1,.1],[.1,.2,.3,.4]])
        np.testing.assert_array_equal(diagnosis.rank_true(p,np.array([1,1,0])),[2,1,4])
    def test_distribution_reports_unrounded_quantiles(self):
        values=diagnosis.distribution([0.,1.,2.,3.])
        self.assertEqual(values['mean'],1.5);self.assertEqual(values['p50'],1.5);self.assertAlmostEqual(values['p90'],2.7)
if __name__=='__main__': unittest.main()
