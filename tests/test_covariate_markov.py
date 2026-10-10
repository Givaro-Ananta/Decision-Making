import unittest
import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from src.models.covariate_markov import objective,fit_transition,predict_log_proba
class TransitionModelTests(unittest.TestCase):
    def test_analytic_gradient(self):
        X=np.c_[np.ones(6),np.array([-2,-1,0,1,2,3])];y=np.array([0,1,2,0,1,2]);theta=np.linspace(-.2,.3,4)
        error=check_grad(lambda b:objective(b,X,y,3)[0],lambda b:objective(b,X,y,3)[1],theta)
        self.assertLess(error,1e-5)
    def test_class_mapping_and_training_only_transform(self):
        x=np.tile([-1.,0.,1.],90);y=np.tile([0,2,3,2,3,0,3,0,2],30)
        frame=pd.DataFrame({'x':x,'next_state':y})
        model,diag=fit_transition(frame,['x'],{'x':0.},{'x':1.})
        self.assertTrue(diag['success']); logs=predict_log_proba(model,frame)
        np.testing.assert_allclose(np.exp(logs).sum(axis=1),1)
        np.testing.assert_array_equal(np.exp(logs)[:,1],0)
        np.testing.assert_allclose(np.exp(logs)[:,[0,2,3]],1/3,atol=1e-6)
        self.assertEqual(model['means'],{'x':0.});self.assertEqual(model['reference_state'],3)
    def test_deterministic(self):
        logs=predict_log_proba({'type':'deterministic','next_state':2},pd.DataFrame(index=range(3)))
        np.testing.assert_array_equal(np.exp(logs),np.tile([0,0,1,0],(3,1)))
if __name__=='__main__': unittest.main()
