import importlib.util
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import check_grad
from src.models.covariate_markov import objective
from src.models.weighted_markov_sensitivity import weighted_objective,class_weights
spec=importlib.util.spec_from_file_location('decisions',Path(__file__).resolve().parents[1]/'scripts/three_state_decision_sensitivity_050.py')
decisions=importlib.util.module_from_spec(spec);spec.loader.exec_module(decisions)
class DecisionSensitivityTests(unittest.TestCase):
    def test_unity_weights_match_original_likelihood(self):
        X=np.c_[np.ones(5),[-2,-1,0,1,2]];y=np.array([0,1,2,0,1]);b=np.array([.1,.2,-.1,.3])
        loss,gradient=weighted_objective(b,X,y,3,np.ones(5));old,grad=objective(b,X,y,3)
        self.assertAlmostEqual(loss,old);np.testing.assert_allclose(gradient,grad)
    def test_weighted_gradient(self):
        X=np.c_[np.ones(5),[-2,-1,0,1,2]];y=np.array([0,1,2,0,1]);weights=np.array([.5,2,3,.5,2]);b=np.array([.1,.2,-.1,.3])
        error=check_grad(lambda x:weighted_objective(x,X,y,3,weights)[0],lambda x:weighted_objective(x,X,y,3,weights)[1],b)
        self.assertLess(error,1e-5)
    def test_balanced_weights_use_training_counts(self):
        y=np.array([0,0,0,0,1,1,2]);cw=class_weights(y,1)
        totals=[sum(cw[int(c)] for c in y if c==j) for j in range(3)]
        np.testing.assert_allclose(totals,[len(y)/3]*3)
    def test_guard_and_two_stage(self):
        p=np.array([[.6,.25,.15],[.3,.2,.5],[.8,.1,.1]])
        np.testing.assert_array_equal(decisions.decide(p,'nonhigh_override',.2),[1,1,0])
        np.testing.assert_array_equal(decisions.decide(p,'nonhigh_guarded',.2),[1,2,0])
        np.testing.assert_array_equal(decisions.decide(p,'two_stage',.3,.5),[1,2,0])
    def test_false_positive_cost_separates_high_errors(self):
        y=np.array([0,1,2]);p=np.array([[.8,.1,.1],[.7,.2,.1],[.1,.1,.8]]);logs=np.log(p)
        m,_,_=decisions.evaluation(y,p,logs,np.array([1,1,1]),np.array([0,0,2]))
        self.assertEqual(m['delta_nonhigh_tp'],1);self.assertEqual(m['delta_nonhigh_fp'],2)
        self.assertEqual(m['extra_nonhigh_fp_per_extra_tp'],2);self.assertEqual(m['delta_no_activity_false_active'],1);self.assertEqual(m['delta_high_tp'],-1)
if __name__=='__main__':unittest.main()
