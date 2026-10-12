import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('sensitivity',Path(__file__).resolve().parents[1]/'scripts/state_definition_sensitivity_050.py')
sensitivity=importlib.util.module_from_spec(spec);spec.loader.exec_module(sensitivity)
class StateSensitivityTests(unittest.TestCase):
    def test_boundary_and_no_activity(self):
        np.testing.assert_array_equal(sensitivity.state3([0,1,1,1],[0,5.96,16.8399,16.84],16.84),[0,1,1,2])
    def test_state_scores_use_three_fixed_labels(self):
        y=np.array([0,1,2]);logs=np.log(np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]))
        m,classes,cm=sensitivity.score(y,logs,3)
        self.assertEqual(m['macro_f1'],1);self.assertEqual(m['balanced_accuracy'],1)
        self.assertEqual(len(classes),3);self.assertEqual(cm.shape,(3,3))
if __name__=='__main__':unittest.main()
