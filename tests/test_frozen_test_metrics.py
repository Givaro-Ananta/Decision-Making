"""Synthetic-only metric checks; never open real final-test data."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('final_metrics',Path(__file__).resolve().parents[1]/'scripts/evaluate_frozen_primary_050_test.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class FinalMetricTests(unittest.TestCase):
    def test_fixed_label_macro_and_weighted_metrics(self):
        y=np.array([0,0,1,2]);p=np.array([[.8,.1,.1],[.8,.1,.1],[.7,.2,.1],[.1,.1,.8]])
        m,classes,cm=module.metrics(y,p,np.log(p),p.argmax(axis=1))
        self.assertAlmostEqual(m['accuracy'],.75);self.assertAlmostEqual(m['balanced_accuracy'],2/3)
        self.assertAlmostEqual(m['macro_f1'],.6);self.assertAlmostEqual(m['weighted_f1'],.65)
        self.assertEqual(cm.sum(),4);self.assertEqual(classes[1]['recall'],0)
    def test_undefined_auc_is_explicit(self):
        p=np.array([[.8,.1,.1],[.9,.05,.05]]);m,classes,cm=module.metrics(np.array([0,0]),p,np.log(p),np.array([0,0]))
        self.assertIsNone(classes[0]['auc_ovr']);self.assertIsNone(classes[1]['auc_ovr'])
        self.assertAlmostEqual(m['macro_f1'],1/3);self.assertEqual(m['balanced_accuracy'],1)
if __name__=='__main__':unittest.main()
