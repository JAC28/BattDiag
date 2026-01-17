"""
Refactor tests - comparison tests with original EvalModupy implementation.
Tests all functions for identical behavior as the original.
"""

import unittest
import numpy as np
from numba.typed import List
from numba import int16
from src.battdiag import cellEval, crossCellEval

# Try to import original implementation for comparison
try:
    import EvalModupy.EvalModupy as EM
    EVALMODUPY_AVAILABLE = True
    print("✓ EvalModupy found - comparison tests will be performed")
except ImportError:
    EVALMODUPY_AVAILABLE = False
    print("⚠ EvalModupy not available - skipping comparison with original implementation")


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
    """Test cellEval module functions - comparison with original"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
    
    def _compare_with_original(self, result_new, result_orig):
        """Compare with original implementation if available"""
        if not EVALMODUPY_AVAILABLE:
            self.skipTest("EvalModupy not available for comparison")
        np.testing.assert_allclose(result_new, result_orig, rtol=1e-6, atol=1e-8, equal_nan=True)
    
    def test_numba_rolling_avgZscore(self):
        """Test rolling average Z-score"""
        result = cellEval.numba_rolling_avgZscore(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.numba_rolling_avgZscore(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_avgdevMean(self):
        """Test rolling average deviation from mean"""
        result = cellEval.numba_rolling_avgdevMean(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.numba_rolling_avgdevMean(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_avgmaxdUdt(self):
        """Test rolling average max dU/dt"""
        result = cellEval.numba_rolling_avgmaxdUdt(
            self.data,
            window=TestConfig.WINDOW,
            dt=TestConfig.DT,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.numba_rolling_avgmaxdUdt(
            self.data,
            window=TestConfig.WINDOW,
            dt=TestConfig.DT,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_rolling_LOF(self):
        """Test rolling Local Outlier Factor"""
        result = cellEval.rolling_LOF(
            self.data,
            window=TestConfig.WINDOW,
            neighbors=TestConfig.NEIGHBORS,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.rolling_LOF(
            self.data,
            window=TestConfig.WINDOW,
            neighbors=TestConfig.NEIGHBORS,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_rolling_LoOP(self):
        """Test rolling Local Outlier Probability"""
        result = cellEval.rolling_LoOP(
            self.data,
            window=TestConfig.WINDOW,
            neighbors=TestConfig.NEIGHBORS,
            extent=TestConfig.EXTENT,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.rolling_LoOP(
            self.data,
            window=TestConfig.WINDOW,
            neighbors=TestConfig.NEIGHBORS,
            extent=TestConfig.EXTENT,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_ShanEn(self):
        """Test rolling Shannon Entropy"""
        result = cellEval.numba_rolling_ShanEn(
            self.data,
            window=TestConfig.WINDOW,
            L=TestConfig.L,
            kind=TestConfig.SHAN_KIND,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.numba_rolling_ShanEn(
            self.data,
            window=TestConfig.WINDOW,
            L=TestConfig.L,
            kind=TestConfig.SHAN_KIND,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_ApEn(self):
        """Test rolling Approximate Entropy"""
        result = cellEval.numba_rolling_ApEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.numba_rolling_ApEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_SampEn(self):
        """Test rolling Sample Entropy"""
        result = cellEval.numba_rolling_SampEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.cellEval.numba_rolling_SampEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)


class TestCrossCellEvalFunctions(unittest.TestCase):
    """Test crossCellEval module functions - comparison with original"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
    
    def _compare_with_original(self, result_new, result_orig):
        """Compare with original implementation if available"""
        if not EVALMODUPY_AVAILABLE:
            self.skipTest("EvalModupy not available for comparison")
        np.testing.assert_allclose(result_new, result_orig, rtol=1e-6, atol=1e-8, equal_nan=True)
    
    def test_numba_rolling_PearCorr(self):
        """Test rolling Pearson Correlation"""
        result = crossCellEval.numba_rolling_PearCorr(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.intraCellEval.numba_rolling_PearCorr(
            self.data,
            window=TestConfig.WINDOW,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_ICC_1(self):
        """Test rolling ICC(1)"""
        result = crossCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(1)",
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.intraCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(1)",
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_ICC_A1(self):
        """Test rolling ICC(A,1)"""
        result = crossCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(A,1)",
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.intraCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(A,1)",
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_ICC_C1(self):
        """Test rolling ICC(C,1)"""
        result = crossCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(C,1)",
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.intraCellEval.numba_rolling_ICC(
            self.data,
            window=TestConfig.WINDOW,
            kind="ICC(C,1)",
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_crossSampEn(self):
        """Test rolling Cross-Sample Entropy"""
        result = crossCellEval.numba_rolling_crossSampEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.intraCellEval.numba_rolling_crossSampEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)
    
    def test_numba_rolling_crossApEn(self):
        """Test rolling Cross-Approximate Entropy"""
        result = crossCellEval.numba_rolling_crossApEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        
        result_orig = EM.utils.intraCellEval.numba_rolling_crossApEn(
            self.data,
            window=TestConfig.WINDOW,
            m=TestConfig.M,
            r=TestConfig.R,
            **TestConfig.get_default_params()
        )
        self._compare_with_original(result, result_orig)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions - comparison with original"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data_2d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
        cls.data_3d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS, TestConfig.N_CELLS)
    
    def _compare_with_original(self, result_new, result_orig):
        """Compare with original implementation if available"""
        if not EVALMODUPY_AVAILABLE:
            self.skipTest("EvalModupy not available for comparison")
        np.testing.assert_allclose(result_new, result_orig, rtol=1e-6, atol=1e-8, equal_nan=True)
    
    def test_numba_2D_zScore(self):
        """Test 2D Z-score normalization"""
        result = cellEval.numba_2D_zScore(self.data_2d, accuracy=8)
        
        result_orig = EM.utils.cellEval.numba_2D_zScore(self.data_2d, accuracy=8)
        self._compare_with_original(result, result_orig)
    
    def test_numba_3D_zScore(self):
        """Test 3D Z-score normalization"""
        result = crossCellEval.numba_3D_zScore(self.data_3d, accuracy=8)
        
        result_orig = EM.utils.intraCellEval.numba_3D_zScore(self.data_3d, accuracy=8)
        self._compare_with_original(result, result_orig)
    
    def test_numba_std_ax0_cellEval(self):
        """Test standard deviation along axis 0 (cellEval)"""
        result = cellEval.numba_std_ax0(self.data_2d)
        
        result_orig = EM.utils.cellEval.numba_std_ax0(self.data_2d)
        self._compare_with_original(result, result_orig)
    
    def test_numba_std_ax0_crossCellEval(self):
        """Test standard deviation along axis 0 (crossCellEval)"""
        result = crossCellEval.numba_std_ax0(self.data_2d)
        
        result_orig = EM.utils.intraCellEval.numba_std_ax0(self.data_2d)
        self._compare_with_original(result, result_orig)
    
    def test_numba_mean_axis0(self):
        """Test mean along axis 0"""
        result = cellEval.numba_mean_axis0(self.data_2d)
        
        result_orig = EM.utils.cellEval.numba_mean_axis0(self.data_2d)
        self._compare_with_original(result, result_orig)
    
    def test_numba_reduceArray_min(self):
        """Test array reduction with min method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "min")
        
        result_orig = EM.utils.cellEval.numba_reduceArray(self.data_2d, int16(fraction), "min")
        self._compare_with_original(result, result_orig)
    
    def test_numba_reduceArray_mean(self):
        """Test array reduction with mean method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "mean")
        
        result_orig = EM.utils.cellEval.numba_reduceArray(self.data_2d, int16(fraction), "mean")
        self._compare_with_original(result, result_orig)
    
    def test_numba_reduceArray_first(self):
        """Test array reduction with first method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "first")
        
        result_orig = EM.utils.cellEval.numba_reduceArray(self.data_2d, int16(fraction), "first")
        self._compare_with_original(result, result_orig)
    
    def test_numba_reduceArray_last(self):
        """Test array reduction with last method"""
        fraction = 10
        result = cellEval.numba_reduceArray(self.data_2d, int16(fraction), "last")
        
        result_orig = EM.utils.cellEval.numba_reduceArray(self.data_2d, int16(fraction), "last")
        self._compare_with_original(result, result_orig)


class TestProcessingFunctions(unittest.TestCase):
    """Test pre-, mid- and post-processing functions - comparison with original"""
    
    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests"""
        np.random.seed(TestConfig.RANDOM_SEED)
        cls.data_2d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS)
        cls.data_3d = np.random.rand(TestConfig.N_SAMPLES, TestConfig.N_CELLS, TestConfig.N_CELLS)
        cls.window = 100
    
    def _compare_with_original(self, result_new, result_orig):
        """Compare with original implementation if available"""
        if not EVALMODUPY_AVAILABLE:
            self.skipTest("EvalModupy not available for comparison")
        np.testing.assert_allclose(result_new, result_orig, rtol=1e-6, atol=1e-8, equal_nan=True)
    
    # Pre-processing tests (cellEval)
    def test_apply_PreProcessing_None_cellEval(self):
        """Test pre-processing with None method (cellEval)"""
        result_data, result_window = cellEval.apply_PreProcessing(
            "None", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        
        result_data_orig, result_window_orig = EM.utils.cellEval.apply_PreProcessing(
            "None", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self._compare_with_original(result_data, result_data_orig)
        self.assertEqual(result_window, result_window_orig)
    
    def test_apply_PreProcessing_MinDownsample_cellEval(self):
        """Test pre-processing with Min-Downsample (cellEval)"""
        result_data, result_window = cellEval.apply_PreProcessing(
            "Min-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        
        result_data_orig, result_window_orig = EM.utils.cellEval.apply_PreProcessing(
            "Min-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self._compare_with_original(result_data, result_data_orig)
        self.assertEqual(result_window, result_window_orig)
    
    def test_apply_PreProcessing_MeanDownsample_cellEval(self):
        """Test pre-processing with Mean-Downsample (cellEval)"""
        result_data, result_window = cellEval.apply_PreProcessing(
            "Mean-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        
        result_data_orig, result_window_orig = EM.utils.cellEval.apply_PreProcessing(
            "Mean-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self._compare_with_original(result_data, result_data_orig)
        self.assertEqual(result_window, result_window_orig)
    
    # Pre-processing tests (crossCellEval)
    def test_apply_PreProcessing_None_crossCellEval(self):
        """Test pre-processing with None method (crossCellEval)"""
        result_data, result_window = crossCellEval.apply_PreProcessing(
            "None", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        
        result_data_orig, result_window_orig = EM.utils.intraCellEval.apply_PreProcessing(
            "None", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self._compare_with_original(result_data, result_data_orig)
        self.assertEqual(result_window, result_window_orig)
    
    def test_apply_PreProcessing_MinDownsample_crossCellEval(self):
        """Test pre-processing with Min-Downsample (crossCellEval)"""
        result_data, result_window = crossCellEval.apply_PreProcessing(
            "Min-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        
        result_data_orig, result_window_orig = EM.utils.intraCellEval.apply_PreProcessing(
            "Min-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self._compare_with_original(result_data, result_data_orig)
        self.assertEqual(result_window, result_window_orig)
    
    def test_apply_PreProcessing_MeanDownsample_crossCellEval(self):
        """Test pre-processing with Mean-Downsample (crossCellEval)"""
        result_data, result_window = crossCellEval.apply_PreProcessing(
            "Mean-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        
        result_data_orig, result_window_orig = EM.utils.intraCellEval.apply_PreProcessing(
            "Mean-Downsample", self.data_2d, int16(self.window), List([TestConfig.FRACTION])
        )
        self._compare_with_original(result_data, result_data_orig)
        self.assertEqual(result_window, result_window_orig)
    
    # Mid-processing tests (cellEval)
    def test_apply_midProcessing_None_cellEval(self):
        """Test mid-processing with None method (cellEval)"""
        result = cellEval.apply_midProcessing(
            "None", self.data_2d, TestConfig.MID_PARAMETERS
        )
        
        result_orig = EM.utils.cellEval.apply_midProcessing(
            "None", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self._compare_with_original(result, result_orig)
    
    def test_apply_midProcessing_Rectangle_cellEval(self):
        """Test mid-processing with Rectangle method (cellEval)"""
        result = cellEval.apply_midProcessing(
            "Rectangle", self.data_2d, TestConfig.MID_PARAMETERS
        )
        
        result_orig = EM.utils.cellEval.apply_midProcessing(
            "Rectangle", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self._compare_with_original(result, result_orig)
    
    # Mid-processing tests (crossCellEval)
    def test_apply_midProcessing_None_crossCellEval(self):
        """Test mid-processing with None method (crossCellEval)"""
        result = crossCellEval.apply_midProcessing(
            "None", self.data_2d, TestConfig.MID_PARAMETERS
        )
        
        result_orig = EM.utils.intraCellEval.apply_midProcessing(
            "None", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self._compare_with_original(result, result_orig)
    
    def test_apply_midProcessing_Rectangle_crossCellEval(self):
        """Test mid-processing with Rectangle method (crossCellEval)"""
        result = crossCellEval.apply_midProcessing(
            "Rectangle", self.data_2d, TestConfig.MID_PARAMETERS
        )
        
        result_orig = EM.utils.intraCellEval.apply_midProcessing(
            "Rectangle", self.data_2d, TestConfig.MID_PARAMETERS
        )
        self._compare_with_original(result, result_orig)
    
    # Post-processing tests (cellEval - 2D)
    def test_apply_PostProcessing_None_cellEval(self):
        """Test post-processing with None method (cellEval)"""
        result = cellEval.apply_PostProcessing(
            "None", self.data_2d, TestConfig.POST_PARAMETERS
        )
        
        result_orig = EM.utils.cellEval.apply_PostProcessing(
            "None", self.data_2d, TestConfig.POST_PARAMETERS
        )
        self._compare_with_original(result, result_orig)
    
    def test_apply_PostProcessing_zScore_cellEval(self):
        """Test post-processing with zScore method (cellEval)"""
        result = cellEval.apply_PostProcessing(
            "zScore", self.data_2d, TestConfig.POST_PARAMETERS
        )
        
        result_orig = EM.utils.cellEval.apply_PostProcessing(
            "zScore", self.data_2d, TestConfig.POST_PARAMETERS
        )
        self._compare_with_original(result, result_orig)
    
    # Post-processing tests (crossCellEval - 3D)
    def test_apply_PostProcessing_None_crossCellEval(self):
        """Test post-processing with None method (crossCellEval)"""
        result = crossCellEval.apply_PostProcessing(
            "None", self.data_3d, TestConfig.POST_PARAMETERS
        )
        
        result_orig = EM.utils.intraCellEval.apply_PostProcessing(
            "None", self.data_3d, TestConfig.POST_PARAMETERS
        )
        self._compare_with_original(result, result_orig)
    
    def test_apply_PostProcessing_zScore_crossCellEval(self):
        """Test post-processing with zScore method (crossCellEval)"""
        result = crossCellEval.apply_PostProcessing(
            "zScore", self.data_3d, TestConfig.POST_PARAMETERS
        )
        
        result_orig = EM.utils.intraCellEval.apply_PostProcessing(
            "zScore", self.data_3d, TestConfig.POST_PARAMETERS
        )
        self._compare_with_original(result, result_orig)


if __name__ == '__main__':
    unittest.main()

