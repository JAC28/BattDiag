#region Import of Packages
import numpy as np
from numba import njit, float64, int16, prange 
from numba.types import string, ListType, Tuple
from numba.typed import List
from sklearn.neighbors import LocalOutlierFactor
from PyNomaly import loop
from numpy.lib.stride_tricks import sliding_window_view


import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning
warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)

#endregion 
if __name__ == '__main__':
   print("This script is not intended to run.")

cache_numba_functions=True

#region Helper-Functions (needed directly in module for Numba compatibility)
@njit(float64[:,:](float64[:,:], int16), parallel=True, fastmath=True, cache=cache_numba_functions)
def numba_2D_zScore(data, accuracy=8):
    """Z-score normalize each time sample independently.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Time-series data where T is samples, N is channels.
    accuracy : int, optional
        Decimal places for rounding. Default 8.
    
    Returns
    -------
    ndarray, shape (T, N)
        Normalized data, rounded to specified accuracy.
    """
    T, N = data.shape
    zScore = np.empty_like(data)
    for t in prange(T):
        zScore[t,:] = np.divide(np.subtract(data[t,:], np.nanmean(data[t,:])), np.nanstd(data[t,:]))
    return np.around(zScore, accuracy)

@njit(float64[:](float64[:,:]), cache=cache_numba_functions)
def numba_std_ax0(X):
    """Compute standard deviation for each column.
    
    Arguments
    ---------
    X : ndarray, shape (T, N)
        Input array where T is rows, N is columns.
    
    Returns
    -------
    ndarray, shape (N,)
        Standard deviation per column.
    """
    result = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        result[i] = np.std(X[:,i])
    return result

@njit(float64[:](float64[:,:]), cache=cache_numba_functions)
def numba_mean_axis0(data):
    """Compute mean for each column.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input array where T is rows, N is columns.
    
    Returns
    -------
    ndarray, shape (N,)
        Mean per column.
    """
    result = np.empty(data.shape[1], dtype=float64)
    for col in range(data.shape[1]):
        result[col] = data[:,col].mean()
    return result

@njit(float64[:,:](float64[:,:], int16, string), parallel=True, fastmath=True, cache=cache_numba_functions)
def numba_reduceArray(array, fraction, method="min"):
    """Downsample by dividing into chunks and applying aggregation method.
    
    Arguments
    ---------
    array : ndarray, shape (T, N)
        Input array to downsample.
    fraction : int
        Downsampling factor; chunks have size T//fraction.
    method : str, optional
        Aggregation method: 'min', 'mean', 'first', or 'last'. Default 'min'.
    
    Returns
    -------
    ndarray, shape (T//fraction, N)
        Downsampled array.
    """
    num_chunks = array.shape[0] // fraction
    chunks = np.array_split(array, num_chunks, axis=0)
    result = np.empty((len(chunks), array.shape[1]))
    for c in prange(len(chunks)):
        for i in range(array.shape[1]):
            if method == "min":
                result[c,i] = np.min(chunks[c][:,i])
            elif method == "mean":
                result[c,i] = np.mean(chunks[c][:,i])
            elif method == "first":
                result[c,i] = chunks[c][0,i]
            elif method == "last":
                result[c,i] = chunks[c][-1,i]
            else:
                ValueError("Given reduction method is invalid.") #Raising the error inside numba parallel loop causes issues
    return result
#endregion

#region Pre-, Data- and Postprocessing
## Pre-processing
@njit(Tuple((float64[:,:], int16))(string, float64[:,:], int16, ListType(float64)), fastmath=True, parallel=False, cache=cache_numba_functions)
def apply_PreProcessing(preProcessing, data, window, parameters):
    """Apply optional downsampling preprocessing.
    
    Arguments
    ---------
    preProcessing : str
        Method: 'None', 'Min-Downsample', or 'Mean-Downsample'.
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size; adjusted by downsampling factor if applied.
    parameters : list of float
        [downsampling_factor] for downsample methods.
    
    Returns
    -------
    tuple of (ndarray, int)
        Filtered data (same shape as input) and adjusted window size.
    """
    if preProcessing =="None":
        return data, window
    elif preProcessing =="Min-Downsample":
        method = "min"
        fraction = int16(parameters[0])
        return numba_reduceArray(data, fraction, method), window//fraction
    elif preProcessing =="Mean-Downsample":
        method = "mean"
        fraction = int16(parameters[0])
        return numba_reduceArray(data, fraction, method), window//fraction
    else:
        raise ValueError("Given preprocessing method is invalid.")

## Additive Rectangle
@njit(float64[:,:](string, float64[:,:],ListType(float64)), fastmath=True, cache=cache_numba_functions)  # Parallel slows down execution
def apply_midProcessing(midProcessing, data, parameters):
    """Apply windowed transformations (additive rectangle) to data.
    
    Arguments
    ---------
    midProcessing : str
        Method: 'None' or 'Rectangle'.
    data : ndarray, shape (T, N)
        Input data to transform.
    parameters : list
        [amplitude, period] for Rectangle method.
    
    Returns
    -------
    ndarray, shape (T, N)
        Transformed data with same shape as input.
    """
    def numba_additiveRectangle(data, A, P):
        rectangles = np.ones_like(data)
        rectangles[np.arange(rectangles.shape[0])%(2*P)>=P,:]=-1
        return data + rectangles*A*np.median(numba_std_ax0(data))
    if midProcessing =="None":
        return data
    elif midProcessing == "Rectangle":
        A = parameters[0]
        P = int16(parameters[1])
        return numba_additiveRectangle(data,A,P)
    else:
        raise ValueError("Given midprocessing method is invalid.")

## Post-processing
@njit(float64[:,:](string, float64[:,:], ListType(float64)), fastmath=True, cache=cache_numba_functions)
def apply_PostProcessing(postProcessing, data, parameters):
    """Normalize or transform data using post-processing methods.
    
    Arguments
    ---------
    postProcessing : str
        Method: 'None' (rounding) or 'zScore' (normalization).
    data : ndarray, shape (T, N)
        Input data to postprocess.
    parameters : list
        [accuracy] - decimal places for rounding (int).
    
    Returns
    -------
    ndarray, shape (T, N)
        Postprocessed data.
    """
    if postProcessing == "None":
        accuracy = int16(parameters[0])
        return np.around(data, accuracy)
    elif postProcessing == "zScore":
        accuracy = int16(parameters[0])
        return numba_2D_zScore(data, accuracy)
    else:
        raise ValueError("Given postprocessing method is invalid.")
#endregion

#region Standard Rolling Function Signature
# ============================================================================
# STANDARD SIGNATURE FOR ALL ROLLING METRIC FUNCTIONS
# ============================================================================
# All rolling_* functions follow this consistent signature:
#
#   def rolling_METRIC_NAME(
#       data,                      # Input data (ndarray)
#       window,                    # Window size (int)
#       # Metric-specific parameters (if any)
#       [optional metric params],  # e.g., neighbors, extent, m, r, L, kind, dt
#       # Standard post-processing parameters
#       accuracy=8,                # Decimal rounding precision
#       # Processing pipeline (consistent ordering)
#       preProcessing="None",      # "None", "Min-Downsample", "Mean-Downsample"
#       preParameters=List([...]),
#       midProcessing="None",      # "None", "Rectangle"
#       midParameters=List([...]),
#       postProcessing="None",     # "None", "zScore"
#       postParameters=List([...]),
#   )
#
# Benefits:
# - Consistent API across all metrics
# - Processing parameters always in same position
# - Metric-specific params grouped at top
# - Easy to extend with new metrics
# ============================================================================
#endregion

#region Numba Functions


@njit(float64[:](float64[:,:], int16), parallel=False, fastmath=True, cache=cache_numba_functions)
def numba_avgZscore(data, accuracy):
    T, N = data.shape
    avgZscores=np.zeros_like(data)
    avgZscore=np.zeros(N)
    for t in range(T):
        mean= np.mean(data[t])
        std = np.around(np.std(data[t]), accuracy)
        for c in range(N):
            avgZscores[t,c]=np.around((mean-data[t,c]),accuracy)/ std
    for c in range(N):
        avgZscore[c]=np.mean(avgZscores[:,c])
    return avgZscore

@njit(float64[:,:](float64[:,:], int16, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True, fastmath=False, cache=cache_numba_functions)#Fastmath disables ability to identify nan
def numba_rolling_avgZscore(data, window, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate average z-score for each sample across cells in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters.
    midProcessing : str, optional
        Mid-processing method. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters.
    postProcessing : str, optional
        Post-processing method. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters.
    
    Returns
    -------
    ndarray, shape (T, N)
        Average z-score per cell at each time point.
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=np.float64)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t] 
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1]=numba_avgZscore(w,accuracy)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

@njit(float64[:](float64[:,:]), parallel=False, fastmath=True, cache=cache_numba_functions)
def numba_avgdevMean(data):
    T, N = data.shape
    devMeans=np.zeros_like(data)
    avgMeans=np.zeros(N)
    for t in range(T):
        mean=np.mean(data[t])
        for c in range(N):
            devMeans[t,c]=mean-data[t,c]
    for c in range(N):
        avgMeans[c]=np.mean(devMeans[:,c])
    return avgMeans

# Methodology based on Gao et al. (2021) DOI: 10.1109/TPEL.2020.3013191 but modified to evaluate all cells
@njit(float64[:,:](float64[:,:], int16, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True, fastmath=False, cache=cache_numba_functions)#Fastmath disables ability to identify nan
def numba_rolling_avgdevMean(data, window, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate average deviation from mean per cell in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method: 'None', 'Min-Downsample', or 'Mean-Downsample'. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters: [downsampling_factor].
    midProcessing : str, optional
        Mid-processing method: 'None' or 'Rectangle'. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters: [amplitude, period].
    postProcessing : str, optional
        Post-processing method: 'None' or 'zScore'. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters: [accuracy].
    
    Returns
    -------
    ndarray, shape (T, N)
        Average deviation per cell at each time point.
    
    References
    ----------
    Gao et al. (2021) DOI: 10.1109/TPEL.2020.3013191
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    # Scale to mV to reduce numerical noise before differencing
    data=data*1000
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=np.float64)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t] 
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1]=numba_avgdevMean(w)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

@njit(float64[:](float64[:,:], float64), parallel=False, fastmath=True, cache=cache_numba_functions)
def numba_avgmaxdUdt(data, dt=0.1):
    T, N = data.shape
    maxdUdt=np.zeros_like(data)
    avgmaxdUdt=np.zeros(N)
    tmp_max=np.zeros(2)
    for t in range(T):
        tmp_max[1]=tmp_max[0]
        tmp_max[0]=np.max(data[t])
        if t==0:
            maxdUdt[t,:]=0
        else:
            for c in range(N):
                maxdUdt[t,c]=((tmp_max[0]-data[t,c])-(tmp_max[1]-data[t-1,c]))/dt
    for c in range(N):
        avgmaxdUdt[c]=np.mean(maxdUdt[:,c])
    return avgmaxdUdt

@njit(float64[:,:](float64[:,:], int16, float64, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True, fastmath=False, cache=cache_numba_functions)#Fastmath disables ability to identify nan
def numba_rolling_avgmaxdUdt(data, window, dt=0.1, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate average maximum voltage rate of change in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    dt : float, optional
        Time step for finite difference calculation. Default 0.1.
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method: 'None', 'Min-Downsample', or 'Mean-Downsample'. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters: [downsampling_factor].
    midProcessing : str, optional
        Mid-processing method: 'None' or 'Rectangle'. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters: [amplitude, period].
    postProcessing : str, optional
        Post-processing method: 'None' or 'zScore'. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters: [accuracy].
    
    Returns
    -------
    ndarray, shape (T, N)
        Average maximum dU/dt per cell at each time point.
    References
    ----------
    Gao et al. (2021) DOI: 10.1109/TPEL.2020.3013191 but modified to use all cells
    """
    window_0 = window
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    scale_dt=window_0/window
    T=data.shape[0]
    N=data.shape[1]
    # Scale to mV to stabilize finite differences
    data=data*1000
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=np.float64)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t] 
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1]=numba_avgmaxdUdt(w,dt*scale_dt)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

# Usage of numba is not possible due to utilisation of Sklearn 
# Usage of multiprocessing does not improve calculation duration

#region Outlier detection
def LOF(data, neighbors, accuracy):
    """Calculate Local Outlier Factor for anomaly detection.

    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    neighbors : int
        Number of neighbors for kNN queries. If larger than sample count, all samples used.
    accuracy : int
        Decimal places for rounding.

    Returns
    -------
    ndarray, shape (N,)
        LOF scores for each channel.
    
    References
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
    """
    clf = LocalOutlierFactor(n_neighbors=neighbors, n_jobs=None, algorithm='kd_tree')  # KD tree is ~10x faster than default
    y_pred=clf.fit_predict(data.T)
    scores = np.around(clf.negative_outlier_factor_*-1, accuracy)
    return scores

def rolling_LOF(data, window, neighbors=10, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate Local Outlier Factor scores in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    neighbors : int, optional
        Number of neighbors for kNN queries. Default 10.
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method: 'None', 'Min-Downsample', or 'Mean-Downsample'. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters: [downsampling_factor].
    midProcessing : str, optional
        Mid-processing method: 'None' or 'Rectangle'. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters: [amplitude, period].
    postProcessing : str, optional
        Post-processing method: 'None' or 'zScore'. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters: [accuracy].
    
    Returns
    -------
    ndarray, shape (T, N)
        LOF anomaly scores per cell at each time point.
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=np.float64)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1] = LOF(w,neighbors, accuracy)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

# Disabling Numba improves calculation speed significantly
def LoOP(data, neighbors, extent, accuracy):
    """Calculate local outlier probability for anomaly detection.


    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    neighbors : int
        Number of neighbors to consider per sample.
    extent : int
        Statistical extent in [1,2,3] (lambda × std from mean).
    accuracy : int
        Decimal places for rounding.

    Returns
    -------
    ndarray, shape (N,)
        LoOP scores for each channel.
    
    References
    ----------
    https://github.com/vc1492a/PyNomaly
    """
    m = loop.LocalOutlierProbability(data.T, n_neighbors=neighbors, extent=extent, use_numba=False).fit()
    scores = np.around(np.array(m.local_outlier_probabilities, dtype=float), accuracy)
    return scores

def rolling_LoOP(data, window, neighbors=10, extent=3, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate Local Outlier Probability scores in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    neighbors : int, optional
        Number of neighbors for kNN queries. Default 10.
    extent : int, optional
        Statistical extent in [1,2,3] (lambda × std from mean). Default 3.
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method: 'None', 'Min-Downsample', or 'Mean-Downsample'. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters: [downsampling_factor].
    midProcessing : str, optional
        Mid-processing method: 'None' or 'Rectangle'. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters: [amplitude, period].
    postProcessing : str, optional
        Post-processing method: 'None' or 'zScore'. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters: [accuracy].
    
    Returns
    -------
    ndarray, shape (T, N)
        LoOP anomaly scores per cell at each time point.
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=np.float64)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1] = LoOP(w,neighbors,extent, accuracy)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows
#endregion



#region Entropy
# Implementation of Shannon Entropy based on Yao et al. (2015) DOI: 10.1016/j.jpowsour.2015.05.090, Table 1
# See also Liu et al. (2018) DOI: 10.3390/en11010136 but published equations and indices are unclear
# Other similar publictaions such as Qiu et al. (2021) DOI: 10.1016/j.est.2021.102852 and 
# Hong et al. (2017) DOI: 10.3390/en10070919 calculate entropy on every point in time instead of rolling window
@njit(float64[:](float64[:,:], int16, string), parallel=True, cache=cache_numba_functions)
def numba_ShanEn(X, L=10, kind='ensemble'):
    """Calculate Shannon entropy for time-series data using binned intervals.

    Arguments
    ---------
    X : ndarray, shape (T, N)
        Time-series data where T is samples, N is channels.
    L : int, optional
        Number of histogram bins. Default 10.
    kind : str, optional
        Entropy type: 'ensemble' or 'local'. Default 'ensemble'.

    Returns
    -------
    ndarray, shape (N,)
        Shannon entropy per channel.
    e bin ranges based on kind
    References
    ----------
    Yao et al. (2015) DOI: 10.1016/j.jpowsour.2015.05.090
    """
    # Calculation of bin bounds
    def _bin_range(X,N,kind):
        _ranges=np.empty((N,2), dtype=float64)
        if kind =='local':
            for i in prange(0,_ranges.shape[0]):
                _ranges[i]=np.array([(X[i,:].min(),(X[i,:].max() ))], dtype=float64)
        elif kind == 'ensemble':
            xmin=X.min()
            xmax=X.max()
            _range=np.array([xmin, xmax],dtype=float64)
            for i in prange(0,_ranges.shape[0]):
                _ranges[i]=_range
        return _ranges
    X=X.T # Transformation of data simplifies access per cell via `axis` argument
    N=X.shape[0]
    assert N>= 0 
    _ranges = _bin_range(X,N, kind)
    p=np.zeros((N,L))
    for c in prange(N):
        hist, bin_edges = np.histogram(X[c], bins=L, range=(_ranges[c][0],_ranges[c][1]))
        p[c]=hist / np.sum(hist)
    H=np.empty(N)
    for i in prange(0,N):
        H[i]=-np.nansum(p[i]*np.log(p[i]))
    return H

@njit(float64[:,:](float64[:,:], int16, int16, string, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True, cache=cache_numba_functions)
def numba_rolling_ShanEn(data, window, L=10, kind="local", accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate Shannon entropy in rolling windows using histogram binning.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    L : int, optional
        Number of histogram bins for entropy calculation. Default 10.
    kind : str, optional
        Entropy type: 'local' (per-cell normalization) or 'ensemble' (global). Default 'local'.
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method: 'None', 'Min-Downsample', or 'Mean-Downsample'. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters: [downsampling_factor].
    midProcessing : str, optional
        Mid-processing method: 'None' or 'Rectangle'. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters: [amplitude, period].
    postProcessing : str, optional
        Post-processing method: 'None' or 'zScore'. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters: [accuracy].
    
    Returns
    -------
    ndarray, shape (T, N)
        Shannon entropy values per cell at each time point.
    
    References
    ----------
    Yao et al. (2015) DOI: 10.1016/j.jpowsour.2015.05.090
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=np.float64)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1]=np.around(numba_ShanEn(w,L=L, kind=kind), accuracy)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

## Approximate Entropy
# For approximate and sample entropy, please also refer to Richman and Moorman (2000) DOI: 10.1152/ajpheart.2000.278.6.H2039
# Numba Implementierung für Code von # https://gist.github.com/DustinAlandzes/a835909ffd15b9927820d175a48dee41
@njit(float64(float64[:], int16, float64), parallel=True, fastmath=True, cache=cache_numba_functions)
def _numba_ApEn(U, m, r):
    N=U.shape[0]
    dim=(int(N-m+1),int(m))
    x1=sliding_window_view(U,m)
    x2=sliding_window_view(U,m+1)
    C=np.empty((2,dim[0]))
    for i in prange(0,dim[0]):
        res =np.zeros(2,dtype=int16)
        for j in range(dim[0]):
            if np.max(np.abs(x1[i]-x1[j])) <= r:
                res[0] += 1
                if i <dim[0]-1 and j<dim[0]-1 and np.max(np.abs(x2[i]-x2[j])) <= r:
                    res[1] +=1
        C[0,i]=res[0] / dim[0] 
        C[1,i]=res[1] / (dim[0]-1) 
    C[1,-1]=1
    return (1/(dim[0]) * np.sum(np.log(C[0])))- (1/(dim[0]-1) * np.sum(np.log(C[1])))

@njit(float64[:](float64[:,:], int16, float64), parallel=False, fastmath=True, cache=cache_numba_functions) 
def numba_ApEn(data, m, r=0.2):
    data=data.T
    N=data.shape[0]
    assert N>= 0 
    _ApEn=np.empty(N)
    if data.shape[1]<=m:
        _ApEn[:]=np.nan
    else:
        for c in prange(N):
            tolerance=r*np.std(data[c])
            _ApEn[c]=_numba_ApEn(data[c], m=m, r=tolerance)
    return _ApEn


@njit(float64[:,:](float64[:,:], int16, int16, float64, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=False, fastmath=False, cache=cache_numba_functions) # Parallel improves speed slighlty 
def numba_rolling_ApEn(data, window, m, r, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate approximate entropy in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    m : int
        Embedding dimension.
    r : float
        Tolerance threshold (fraction of std).
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters.
    midProcessing : str, optional
        Mid-processing method. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters.
    postProcessing : str, optional
        Post-processing method. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters.
    
    Returns
    -------
    ndarray, shape (T, N)
        Approximate entropy per cell at each time point.
    
    References
    ----------
    Richman and Moorman (2000) DOI: 10.1152/ajpheart.2000.278.6.H2039
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=float)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1]=numba_ApEn(w, m, r)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

## Sample Entropy
# Code modified from antropy package (https://github.com/raphaelvallat/antropy)
@njit(float64(float64[:], int16, float64), parallel=False, fastmath=True, cache=cache_numba_functions)
def _numba_sampen(sequence, order, r):
    """Fast sample entropy calculation using Numba optimization."""

    size = sequence.size

    numerator = 0
    denominator = 0

    for offset in prange(1, size - order):
        n_numerator = int(abs(sequence[order] - sequence[order + offset]) >= r)
        n_denominator = 0

        for idx in range(order):
            n_numerator += abs(sequence[idx] - sequence[idx + offset]) >= r
            n_denominator += abs(sequence[idx] - sequence[idx + offset]) >= r

        if n_numerator == 0:
            numerator += 1
        if n_denominator == 0:
            denominator += 1

        prev_in_diff = int(abs(sequence[order] - sequence[offset + order]) >= r)
        for idx in range(1, size - offset - order):
            out_diff = int(abs(sequence[idx - 1] - sequence[idx + offset - 1]) >= r)
            in_diff = int(abs(sequence[idx + order] - sequence[idx + offset + order]) >= r)
            n_numerator += in_diff - out_diff
            n_denominator += prev_in_diff - out_diff
            prev_in_diff = in_diff

            if n_numerator == 0:
                numerator += 1
            if n_denominator == 0:
                denominator += 1

    if denominator == 0 or numerator == 0:
        return np.log(size-order)+np.log(size-order-1)+np.log(2)  # Maximum value as defined by Richman & Moorman (2000)
    else:
        return -np.log(numerator / denominator)

@njit(float64[:](float64[:,:], int16, float64), parallel=True, fastmath=True, cache=cache_numba_functions)
def SampEn(data, m, r=0.2):
    data=data.T    
    N=data.shape[0]
    assert N>= 0 
    _SampEn=np.zeros(N)
    if data.shape[1]<=m:
        _SampEn[:]=np.nan
    else:
        for c in prange(N):
            tolerance=r*np.std(data[c])
            _SampEn[c]=_numba_sampen(data[c], order=m, r=tolerance)
    return _SampEn


@njit(float64[:,:](float64[:,:], int16, int16, float64, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=False, cache=cache_numba_functions) # Parallel improves speed slighlty 
def numba_rolling_SampEn(data, window, m, r, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate sample entropy in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    m : int
        Embedding dimension.
    r : float
        Tolerance threshold (fraction of std).
    accuracy : int, optional
        Decimal rounding precision. Default 8.
    preProcessing : str, optional
        Preprocessing method: 'None', 'Min-Downsample', or 'Mean-Downsample'. Default 'None'.
    preParameters : list of float, optional
        Preprocessing parameters: [downsampling_factor].
    midProcessing : str, optional
        Mid-processing method: 'None' or 'Rectangle'. Default 'None'.
    midParameters : list of float, optional
        Mid-processing parameters: [amplitude, period].
    postProcessing : str, optional
        Post-processing method: 'None' or 'zScore'. Default 'None'.
    postParameters : list of float, optional
        Post-processing parameters: [accuracy].
    
    Returns
    -------
    ndarray, shape (T, N)
        Sample entropy per cell at each time point.
    
    References
    ----------
    Richman and Moorman (2000) DOI: 10.1152/ajpheart.2000.278.6.H2039
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N), dtype=float)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        windows[t+window-1]=SampEn(w, m, r)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

#endregion 
