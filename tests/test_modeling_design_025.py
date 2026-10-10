"""Regression checks for estimability diagnostics; no predictive models are fitted."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('design',Path(__file__).resolve().parents[1]/'scripts/finalize_modeling_design_025.py')
design=importlib.util.module_from_spec(spec); spec.loader.exec_module(design)
class SeparationTests(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(design.separation(np.c_[np.ones(4),[-2,-1,1,2]],np.array([0,0,1,1])),'complete')
    def test_quasi(self):
        self.assertEqual(design.separation(np.c_[np.ones(4),[-1,0,0,1]],np.array([0,0,1,1])),'quasi')
    def test_overlap(self):
        self.assertEqual(design.separation(np.c_[np.ones(4),[-1,-1,1,1]],np.array([0,1,0,1])),'none')
    def test_multiclass_overlap(self):
        self.assertEqual(design.separation(np.c_[np.ones(6),[-1,-1,-1,1,1,1]],np.array([0,1,2,0,1,2])),'none')
class GridResolutionTests(unittest.TestCase):
    def test_default_ids_and_nesting(self):
        from src.preprocessing.modeling_grid_025 import cell_id, point_indices
        self.assertEqual(cell_id(2,3), 'SUM025_R002_C003')
        self.assertEqual(cell_id(2,3,.5), 'SUM050_R002_C003')
        lon=np.array([95.01,95.22,95.44,95.71])
        lat=np.array([5.99,5.78,5.56,5.22])
        r25,c25=point_indices(lon,lat)
        r50,c50=point_indices(lon,lat,.5)
        np.testing.assert_array_equal(r50,r25//2)
        np.testing.assert_array_equal(c50,c25//2)
if __name__=='__main__': unittest.main()
