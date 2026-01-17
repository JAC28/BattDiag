#region Import of Packages
import numpy as np
from numba import njit, float64, int16, prange 
from numba.types import string, ListType, Tuple
from numba.typed import List
from antropy import sample_entropy
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
    """Compute z-score normalization for 2D time-series data.
    
    Normalizes each time sample independently by subtracting the mean and
    dividing by standard deviation. Useful for removing offset and scale
    variations in multi-channel measurements.
    
    Parameters
    ----------
    data : ndarray, shape (T, N)
        2D array containing time-series data where T is the number of time
        samples and N is the number of channels/cells.
    accuracy : int, optional
        Number of decimal places to round the result. Default is 8.
    
    Returns
    -------
    ndarray, shape (T, N)
        Z-score normalized data with values rounded to specified accuracy.
        NaN values in input are preserved.
    
    Notes
    -----
    - Numba-optimized with parallelization across time dimension
    - Uses fastmath mode for increased performance (may reduce precision)
    - Handles NaN values by computing statistics while ignoring NaN
    - Each time sample is normalized independently (row-wise normalization)
    """
    T, N = data.shape
    zScore = np.empty_like(data)
    for t in prange(T):
        zScore[t,:] = np.divide(np.subtract(data[t,:], np.nanmean(data[t,:])), np.nanstd(data[t,:]))
    return np.around(zScore, accuracy)

@njit(float64[:](float64[:,:]), cache=cache_numba_functions)
def numba_std_ax0(X):
    """Standard deviation along axis 0"""
    result = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        result[i] = np.std(X[:,i])
    return result

@njit(float64[:](float64[:,:]), cache=cache_numba_functions)
def numba_mean_axis0(data):
    """Mean along axis 0"""
    result = np.empty(data.shape[1], dtype=float64)
    for col in range(data.shape[1]):
        result[col] = data[:,col].mean()
    return result

@njit(float64[:,:](float64[:,:], int16, string), parallel=True, fastmath=True, cache=cache_numba_functions)
def numba_reduceArray(array, fraction, method="min"):
    """Reduce array by chunking and aggregating"""
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
@njit(float64[:,:](string, float64[:,:],ListType(float64)), fastmath=True, cache=cache_numba_functions) # Parallel verlangsamt Ausführung 
def apply_midProcessing(midProcessing, data, parameters): # Nested dunction is better than calling an external function
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
@njit(float64[:,:](float64[:,:], int16, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True, fastmath=False, cache=cache_numba_functions)#Fastmath disables ability to identify nan
def numba_rolling_avgdevMean(data, window, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
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
    """Function for creation of LOF based on sklearn.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor

    Args:
        data (ndarray): Data with shape (T,N)
        neighbors (int): Number of neighbors to use by default for kneighbors queries. If n_neighbors is larger than the number of samples provided, all samples will be used
        accuracy (int): Number of decimal used by numpy.round to minimize effect of etol

    Returns:
        ndarray: Calculated LOF for each n in N at every t in T
    """
    clf = LocalOutlierFactor(n_neighbors=neighbors, n_jobs=None, algorithm='kd_tree') # KD Tree significantly faster with sample data x10
    y_pred=clf.fit_predict(data.T)
    scores = np.around(clf.negative_outlier_factor_*-1, accuracy)
    return scores

def rolling_LOF(data, window, neighbors=10, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Function that applies LOF to a rolling window along the first dimension

    Args:
        data (ndarray): Data with shape (T,N)
        window (int): Window size for calculation along T
        neighbors (int, optional): Number of neighbors to use by default for kneighbors queries. Defaults to 10.
        accuracy (int, optional): Number of decimal used for rounding the result. Defaults to 8.
        rect (bool, optional): Flag to activate the superimposed rectangle. Defaults to False.
        A (float, optional): Amplitude of the superimposed rectangle if "rect" is True. Defaults to 0.1.
        P (int, optional): Period of the superimposed rectangle. Defaults to 2.

    Returns:
        ndarray: LOF-values for all n in N at each t in T calculated based on the last window values
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

# Deaktivierung von Numba führt zu deutlichem Geschwindigkeitsgewinn
def LoOP(data, neighbors, extent, accuracy):
    """Implementaion of Local outlier probability using (https://github.com/vc1492a/PyNomaly)


    Args:
        data (ndarray): Data with shape (T,N)
        neighbors (int): Total number of neighbors to consider w.r.t. each sample
        extent (int): Integer [1,2,3] that controls the statistical extent, e.g. lambda times the standard deviation from the mean
        accuracy (int): Number of decimal used by numpy.round to minimize effect of etol

    Returns:
        ndarray: LOoP-values for all n in N at each t in T calculated based on the last window values
    """
    m = loop.LocalOutlierProbability(data.T, n_neighbors=neighbors, extent=extent, use_numba=False).fit()
    scores = np.around(np.array(m.local_outlier_probabilities, dtype=float), accuracy)
    return scores

def rolling_LoOP(data, window, neighbors=10, extent=3, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Function that applies LoOP to a rolling window along the first dimension

    Args:
        data (ndarray): Data with shape (T,N)
        window (int): Window size for calculation along T
        neighbors (int, optional): Number of neighbors to use by default for kneighbors queries. Defaults to 10.
        extent (int): Integer [1,2,3] that controls the statistical extent, e.g. lambda times the standard deviation from the mean. Defaults to 3.
        accuracy (int, optional): Number of decimal used for rounding the result. Defaults to 8.
        rect (bool, optional): Flag to activate the superimposed rectangle. Defaults to False.
        A (float, optional): Amplitude of the superimposed rectangle if "rect" is True. Defaults to 0.1.
        P (int, optional): Period of the superimposed rectangle. Defaults to 2.

    Returns:
        ndarray: LOoP-values for all n in N at each t in T calculated based on the last window values
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
# Implementation of Shannon Entropy based on Yao.2015 
@njit(float64[:](float64[:,:], int16, string), parallel=True, cache=cache_numba_functions)
def numba_ShanEn(X, L=10, kind='ensemble'):
    """Function for calculation of Shannon Entropy of TimeSeries with 
    binned intervalls. Equations for local and ensemble entropy based on Yao.2015

    Args:
        X (ndarray): 2D array containing the time-series data. Expected shape is 
                     (T,N), where T and N represents the total time and number 
                     of parallel channels, respectively. 
        L (int, optional): Number of bins that is used for seperation of data. Default is 10.
        kind (str, optional): Defines type of Shannon Entropy. Either `ensemble` or `local`
                              are valid names. Default is `ensemble`.

    Returns:
        ndarray : 1D array containing the Entropy values. Shape of result is (N,)
                  where N represents the number of parallel channels. 
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
    """Function for rolling application of ShannonEntropy to dataset

    Note: The plan to define a function that generates the windows first 
    and then calculate the Entropy for each window was deprecated due to 
    significant slower computation when tested with mean() as benchmark. 

    Args:
        data (ndarray): 2D array containing the time-series data. Expected shape is 
                         (T,N), where T and N represents the total time and number 
                         of parallel channels, respectively. 
        window (int): Lenght of observer window. 
        L (int, optional): Number of bins that is used for seperation of data. Default is 10. 
        kind (str, optional): Defines type of Shannon Entropy. Either `ensemble` or `local`
                              are valid names. Default is `local`.
        rect (boolean, optional): Flag to control whether the data is superimposed by a rectangle
                                  signal. Default is False.
        A (float, optional): Amplitude of rectangle signal relative to median of STD. Default is 0.1.
        P (int, optional): Number of periods of rectangle signal added to the signal. Signal is distributed
                           over the given sample size. Default is 2.

    Returns:
        ndarray: 2D array containing the Entropy values at each point in time. Shape of result is (T,N)
                  where T and N represents the total time and number of parallel channels, respectively.
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
def SampEn_old(data, m, r=0.2):
    data=data.T
    data=np.array([row[~np.isnan(row)] for row in data])
    N=data.shape[0]
    assert N>= 0 
    tolerances=r*np.std(data, axis=1)
    _SampEn=np.empty(N)
    if data.shape[1]<=m:
        _SampEn[:]=np.nan
    else:
        for c in prange(N):
            tolerance=tolerances[c]
            _SampEn[c]=sample_entropy(data[c], order=m)
    return _SampEn

@njit(float64(float64[:], int16, float64), parallel=False, fastmath=True, cache=cache_numba_functions) # Paralell improves speed by factor 4x
def _numba_sampen(sequence, order, r):
    """
    Code slighly modified from antropy package.
    Fast evaluation of the sample entropy using Numba.
    """

    size = sequence.size
    # sequence = sequence.tolist()

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
        return np.log(size-order)+np.log(size-order-1)+np.log(2) # Maximum defined by 10.1152/ajpheart.2000.278.6.H2039
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
