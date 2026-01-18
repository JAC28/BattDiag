"""Functional test suite for battdiag package.
Tests all functions for shape and finite values.
"""

import unittest
import numpy as np
from numba.typed import List
from numba import int16
from battdiag import cellEval, crossCellEval

print("✓ Running functional tests (no lokal EvalModupy dependency)")


# Centralized test parameters
class TestConfig:
    """Central configuration for test parameters"""
    WINDOW = 100
    ACCURACY = 8
    FRACTION = 3.0
    PRE_PROCESSING = "None"
    PRE_PARAMETERS = List([FRACTION])
    MID_PROCESSING = "None"
    MID_PARAMETERS = List([0.1, 2.0])
    POST_PROCESSING = "None"
    POST_PARAMETERS = List([8.0])
    
    # Test data sizes
    N_SAMPLES = 500
    N_CELLS = 12
    RANDOM_SEED = 42
    
    # Additional function-specific parameters
    DT = 0.1  # for avgmaxdUdt
    NEIGHBORS = 10  # for LOF, LoOP
    EXTENT = 3  # for LoOP
    L = 10  # for ShanEn
    SHAN_KIND = "local"  # for ShanEn
    M = 2  # for ApEn, SampEn
    R = 0.2  # for ApEn, SampEn
    
    @classmethod
    def get_default_params(cls):
        """Return dictionary with default processing parameters"""
        return {
            'accuracy': cls.ACCURACY,
            'preProcessing': cls.PRE_PROCESSING,
            'preParameters': cls.PRE_PARAMETERS,
            'midProcessing': cls.MID_PROCESSING,
            'midParameters': cls.MID_PARAMETERS,
            'postProcessing': cls.POST_PROCESSING,
            'postParameters': cls.POST_PARAMETERS
        }


class TestCellEvalFunctions(unittest.TestCase):
    """Test cellEval module functions - functional tests"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
    
    def _check_result(self, result, expected_shape):
        """Common checks for all results"""
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_numba_rolling_avgZscore(self):
        """Test rolling average Z-score"""
        result = cellEval.numba_rolling_avgZscore(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)
    
    def test_numba_rolling_avgdevMean(self):
        """Test rolling average deviation from mean"""
        result = cellEval.numba_rolling_avgdevMean(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)
    
    def test_numba_rolling_avgmaxdUdt(self):
        """Test rolling average max dU/dt"""
        result = cellEval.numba_rolling_avgmaxdUdt(
            self.data,
            window=TestConfig.WINDOW,
            dt=TestConfig.DT,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)
    
    def test_rolling_LOF(self):
        """Test rolling Local Outlier Factor"""
        result = cellEval.rolling_LOF(
            self.data,
            window=TestConfig.WINDOW,
            neighbors=TestConfig.NEIGHBORS,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)
    
    def test_rolling_LoOP(self):
        """Test rolling Local Outlier Probability"""
        result = cellEval.rolling_LoOP(
            self.data,
            window=TestConfig.WINDOW,
            neighbors=TestConfig.NEIGHBORS,
            extent=TestConfig.EXTENT,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)
    
    def test_numba_rolling_ShanEn(self):
        """Test rolling Shannon Entropy"""
        result = cellEval.numba_rolling_ShanEn(
            self.data,
            window=TestConfig.WINDOW,
            L=TestConfig.L,
            kind=TestConfig.SHAN_KIND,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)
    
    def test_numba_rolling_ApEn(self):
        """Test rolling Approximate Entropy"""
        result = cellEval.numba_rolling_ApEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)
    
    def test_numba_rolling_SampEn(self):
        """Test rolling Sample Entropy"""
        result = cellEval.numba_rolling_SampEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        self._check_result(result, self.data.shape)


class TestCrossCellEvalFunctions(unittest.TestCase):
    """Test crossCellEval module functions - functional tests"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
    
    def _check_result_3d(self, result, expected_shape):
        """Common checks for 3D results"""
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_numba_rolling_PearCorr(self):
        """Test rolling Pearson Correlation"""
        result = crossCellEval.numba_rolling_PearCorr(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        expected_shape = (self.data.shape[0], self.data.shape[1], self.data.shape[1])
        self._check_result_3d(result, expected_shape)
    
    def test_numba_rolling_ICC_1(self):
        """Test rolling ICC(1)"""
        result = crossCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(1)",
            **TestConfig.get_default_params()
        )
        expected_shape = (self.data.shape[0], self.data.shape[1], self.data.shape[1])
        self._check_result_3d(result, expected_shape)
    
    def test_numba_rolling_ICC_A1(self):
        """Test rolling ICC(A,1)"""
        result = crossCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(A,1)",
            **TestConfig.get_default_params()
        )
        expected_shape = (self.data.shape[0], self.data.shape[1], self.data.shape[1])
        self._check_result_3d(result, expected_shape)
    
    def test_numba_rolling_ICC_C1(self):
        """Test rolling ICC(C,1)"""
        result = crossCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(C,1)",
            **TestConfig.get_default_params()
        )
        expected_shape = (self.data.shape[0], self.data.shape[1], self.data.shape[1])
        self._check_result_3d(result, expected_shape)
    
    def test_numba_rolling_crossSampEn(self):
        """Test rolling Cross-Sample Entropy"""
        result = crossCellEval.numba_rolling_crossSampEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        expected_shape = (self.data.shape[0], self.data.shape[1], self.data.shape[1])
        self._check_result_3d(result, expected_shape)
    
    def test_numba_rolling_crossApEn(self):
        """Test rolling Cross-Approximate Entropy"""
        result = crossCellEval.numba_rolling_crossApEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        expected_shape = (self.data.shape[0], self.data.shape[1], self.data.shape[1])
        self._check_result_3d(result, expected_shape)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions from both modules"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data_2d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
        cls.data_3d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS, TestConfig.N_CELLS)
    
    def test_numba_2D_zScore(self):
        """Test 2D Z-score normalization"""
        result = cellEval.numba_2D_zScore(self.data_2d, accuracy=8)
        self.assertEqual(result.shape, self.data_2d.shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_numba_3D_zScore(self):
        """Test 3D Z-score normalization"""
        result = crossCellEval.numba_3D_zScore(self.data_3d, accuracy=8)
        self.assertEqual(result.shape, self.data_3d.shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_numba_std_ax0_cellEval(self):
        """Test standard deviation along axis 0 (cellEval)"""
        result = cellEval.numba_std_ax0(self.data_2d)
        self.assertEqual(result.shape, (TestConfig.N_CELLS,))
        self.assertTrue(np.all(result >= 0))  # std should be non-negative
    
    def test_numba_std_ax0_crossCellEval(self):
        """Test standard deviation along axis 0 (crossCellEval)"""
        result = crossCellEval.numba_std_ax0(self.data_2d)
        self.assertEqual(result.shape, (TestConfig.N_CELLS,))
        self.assertTrue(np.all(result >= 0))
    
    def test_numba_mean_axis0(self):
        """Test mean along axis 0"""
        result = cellEval.numba_mean_axis0(self.data_2d)
        self.assertEqual(result.shape, (TestConfig.N_CELLS,))
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_numba_reduceArray_min(self):
        """Test array reduction with min method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "min")
        expected_rows = TestConfig.N_SAMPLES // fraction
        self.assertEqual(result.shape, (expected_rows, TestConfig.N_CELLS))
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_numba_reduceArray_mean(self):
        """Test array reduction with mean method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "mean")
        expected_rows = TestConfig.N_SAMPLES // fraction
        self.assertEqual(result.shape, (expected_rows, TestConfig.N_CELLS))
    
    def test_numba_reduceArray_first(self):
        """Test array reduction with first method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "first")
        expected_rows = TestConfig.N_SAMPLES // fraction
        self.assertEqual(result.shape, (expected_rows, TestConfig.N_CELLS))
    
    def test_numba_reduceArray_last(self):
        """Test array reduction with last method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "last")
        expected_rows = TestConfig.N_SAMPLES // fraction
        self.assertEqual(result.shape, (expected_rows, TestConfig.N_CELLS))


class TestProcessingFunctions(unittest.TestCase):
    """Test pre-, mid- and post-processing functions"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data_2d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
        cls.data_3d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS, TestConfig.N_CELLS)
        cls.window = 100
    
    # Pre-processing tests (cellEval)
    def test_apply_PreProcessing_None_cellEval(self):
        """Test pre-processing with None method (cellEval)"""
        result_data, result_window = cellEval.apply_PreProcessing(
            "None", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self.assertEqual(result_data.shape, self.data_2d.shape)
        self.assertEqual(result_window, self.window)
    
    def test_apply_PreProcessing_MinDownsample_cellEval(self):
        """Test pre-processing with Min-Downsample (cellEval)"""
        result_data, result_window = cellEval.apply_PreProcessing(
            "Min-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        expected_rows = TestConfig.N_SAMPLES // int(TestConfig.FRACTION)
        self.assertEqual(result_data.shape, (expected_rows, TestConfig.N_CELLS))
        self.assertEqual(result_window, self.window // int(TestConfig.FRACTION))
    
    def test_apply_PreProcessing_MeanDownsample_cellEval(self):
        """Test pre-processing with Mean-Downsample (cellEval)"""
        result_data, result_window = cellEval.apply_PreProcessing(
            "Mean-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        expected_rows = TestConfig.N_SAMPLES // int(TestConfig.FRACTION)
        self.assertEqual(result_data.shape, (expected_rows, TestConfig.N_CELLS))
        self.assertEqual(result_window, self.window // int(TestConfig.FRACTION))
    
    # Pre-processing tests (crossCellEval)
    def test_apply_PreProcessing_None_crossCellEval(self):
        """Test pre-processing with None method (crossCellEval)"""
        result_data, result_window = crossCellEval.apply_PreProcessing(
            "None", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self.assertEqual(result_data.shape, self.data_2d.shape)
        self.assertEqual(result_window, self.window)
    
    def test_apply_PreProcessing_MinDownsample_crossCellEval(self):
        """Test pre-processing with Min-Downsample (crossCellEval)"""
        result_data, result_window = crossCellEval.apply_PreProcessing(
            "Min-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        expected_rows = TestConfig.N_SAMPLES // int(TestConfig.FRACTION)
        self.assertEqual(result_data.shape, (expected_rows, TestConfig.N_CELLS))
        self.assertEqual(result_window, self.window // int(TestConfig.FRACTION))
    
    def test_apply_PreProcessing_MeanDownsample_crossCellEval(self):
        """Test pre-processing with Mean-Downsample (crossCellEval)"""
        result_data, result_window = crossCellEval.apply_PreProcessing(
            "Mean-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        expected_rows = TestConfig.N_SAMPLES // int(TestConfig.FRACTION)
        self.assertEqual(result_data.shape, (expected_rows, TestConfig.N_CELLS))
        self.assertEqual(result_window, self.window // int(TestConfig.FRACTION))
    
    # Mid-processing tests (cellEval)
    def test_apply_midProcessing_None_cellEval(self):
        """Test mid-processing with None method (cellEval)"""
        result = cellEval.apply_midProcessing(
            "None", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_2d.shape)
        np.testing.assert_array_equal(result, self.data_2d)
    
    def test_apply_midProcessing_Rectangle_cellEval(self):
        """Test mid-processing with Rectangle method (cellEval)"""
        result = cellEval.apply_midProcessing(
            "Rectangle", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_2d.shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    # Mid-processing tests (crossCellEval)
    def test_apply_midProcessing_None_crossCellEval(self):
        """Test mid-processing with None method (crossCellEval)"""
        result = crossCellEval.apply_midProcessing(
            "None", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_2d.shape)
        np.testing.assert_array_equal(result, self.data_2d)
    
    def test_apply_midProcessing_Rectangle_crossCellEval(self):
        """Test mid-processing with Rectangle method (crossCellEval)"""
        result = crossCellEval.apply_midProcessing(
            "Rectangle", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_2d.shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    # Post-processing tests (cellEval - 2D)
    def test_apply_PostProcessing_None_cellEval(self):
        """Test post-processing with None method (cellEval)"""
        result = cellEval.apply_PostProcessing(
            "None", self.data_2d, TestConfig.POST_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_2d.shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_apply_PostProcessing_zScore_cellEval(self):
        """Test post-processing with zScore method (cellEval)"""
        result = cellEval.apply_PostProcessing(
            "zScore", self.data_2d, TestConfig.POST_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_2d.shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    # Post-processing tests (crossCellEval - 3D)
    def test_apply_PostProcessing_None_crossCellEval(self):
        """Test post-processing with None method (crossCellEval)"""
        result = crossCellEval.apply_PostProcessing(
            "None", self.data_3d, TestConfig.POST_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_3d.shape)
        self.assertTrue(np.any(np.isfinite(result)))
    
    def test_apply_PostProcessing_zScore_crossCellEval(self):
        """Test post-processing with zScore method (crossCellEval)"""
        result = crossCellEval.apply_PostProcessing(
            "zScore", self.data_3d, TestConfig.POST_PARAMETERS
        )
        self.assertEqual(result.shape, self.data_3d.shape)
        self.assertTrue(np.any(np.isfinite(result)))


if __name__ == '__main__':
    unittest.main()
