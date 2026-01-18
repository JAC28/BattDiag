#region Import of Packages
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from numba import njit, float64, int16, prange 
from numba.types import string, ListType, Tuple
from numba.typed import List


import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning
warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)

#endregion

#region Helper-Functions
## Standard deviation with axis
@njit(float64[:](float64[:,:]), cache=True)
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

## Numba implementation of zScore over column axes
@njit(float64[:,:,:](float64[:,:,:], int16), parallel=True, fastmath=True, cache=True)
def numba_3D_zScore(data,accuracy=8):
    """Z-score normalize each 2D slice independently.
    
    Arguments
    ---------
    data : ndarray, shape (T, N, N)
        3D array where T is samples, N×N is correlation matrix.
    accuracy : int, optional
        Decimal places for rounding. Default 8.
    
    Returns
    -------
    ndarray, shape (T, N, N)
        Normalized 3D array, rounded to specified accuracy.
    """
    T,N,N = data.shape
    zScore=np.empty_like(data)
    for t in prange(T):
        zScore[t,:,:] = np.divide(np.subtract(data[t,:,:],np.nanmean(data[t,:,:])), np.nanstd(data[t,:,:]))
    return np.around(zScore, accuracy)

## Function downsamples data by fraction and returns updated window
@njit(float64[:,:](float64[:,:], int16, string), parallel=True, fastmath=True, cache=True)
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
                ValueError("Given reduction method is invalid.")  # Raising error inside numba parallel loop causes issues
    return result
#endregion

#region Pre-, Data- and Postprocessing
## Pre-processing
@njit(Tuple((float64[:,:], int16))(string, float64[:,:], int16, ListType(float64)), fastmath=True, parallel=False, cache=True)
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
@njit(float64[:,:](string, float64[:,:],ListType(float64)), fastmath=True, cache=True)  # Parallel slows down execution 
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
@njit(float64[:,:,:](string, float64[:,:,:], ListType(float64)), fastmath=True, cache=True)
def apply_PostProcessing(postProcessing, data, parameters):
    """Normalize or transform 3D data using post-processing methods.
    
    Arguments
    ---------
    postProcessing : str
        Method: 'None' (rounding) or 'zScore' (normalization).
    data : ndarray, shape (T, N, N)
        3D input data to postprocess.
    parameters : list
        [accuracy] - decimal places for rounding (int).
    
    Returns
    -------
    ndarray, shape (T, N, N)
        Postprocessed data.
    """
    if postProcessing == "None":
        accuracy = int16(parameters[0])
        return np.around(data, accuracy)
    elif postProcessing == "zScore":
        accuracy = int16(parameters[0])
        return numba_3D_zScore(data, accuracy)
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
#       [optional metric params],  # e.g., kind, m, r, L, etc.
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
# - Metric-specific params grouped before accuracy
# - Easy to extend with new metrics
# ============================================================================
#endregion

#region Correlation
## Pearson Correlation
@njit(float64[:,:,:](float64[:,:], int16, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True) 
def numba_rolling_PearCorr(data, window, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate Pearson correlation matrix in rolling windows.
    
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
    ndarray, shape (T, N, N)
        Pearson correlation matrix at each time point.
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert T>=0 and N>=0 and window >=0
    assert window >2, "Window has to be larger than 2. Please check Pre-Processing and window size."
    windows=np.empty((T, N,N), dtype=float64)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :,:]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        if w.shape[0]<=1:
            windows[t+window-1,:,:] = np.nan
        else:
            windows[t+window-1]=np.corrcoef(w, rowvar=False)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

## Intraclass Correlation 
# Highly recommend Liljequist et al. (2019) DOI: 10.1371/journal.pone.0219854 as overview of ICC types due to the explanation for non-specialists
# Li et al. (2018) DOI: 10.1016/j.measurement.2017.11.034 claim to use the ICC as well but are not in line with the established definitions of ICC 
# but seem to calculate a Pearson correlation instead. 
# Important note: Despite other statements in literature, ICC is not bounded between 0 and 1 but can also be negative!
@njit(float64(float64[:,:]))
def ICC_1(X):
    """Calculate ICC(1) - one-way random effects model.

    Arguments
    ---------
    X : ndarray, shape (T, N)
        Data where T represents subjects and N represents raters.

    Returns
    -------
    float
        ICC(1) value.
    
    References
    ----------
    Liljequist et al. (2019) DOI: 10.1371/journal.pone.0219854
    """    
    n=X.shape[0]
    k=X.shape[1]
    Xmean=np.mean(X)
    S=1/k*np.sum(X, axis=1)
    SSBS=np.sum(np.array([(S[i]-Xmean)**2 for i in range(0,n) for j in range(0,k)]))
    SSWS=np.sum(np.array([(X[i,j]-S[i])**2 for i in range(0,n) for j in range(0,k)]))
    MSBS=SSBS/(n-1)
    MSWS=SSWS/(n*(k-1))
    return (MSBS-MSWS)/(MSBS+(k-1)*MSWS)

@njit(float64(float64[:,:]))
def ICC_C1(X):
    """Calculate ICC(C,1) - two-way random effects, consistency model.

    Arguments
    ---------
    X : ndarray, shape (T, N)
        Data where T represents subjects and N represents raters.

    Returns
    -------
    float
        ICC(C,1) value.
    
    References
    ----------
    Liljequist et al. (2019) DOI: 10.1371/journal.pone.0219854
    """  
    n=X.shape[0]
    k=X.shape[1]
    Xmean=np.mean(X)
    S=1/k*np.sum(X, axis=1)
    M=1/n*np.sum(X, axis=0)
    SST=np.sum((X-Xmean)**2)
    SSBS=np.sum(np.array([(S[i]-Xmean)**2 for i in range(0,n) for j in range(0,k)]))
    SSBM=np.sum(np.array([(M[j]-Xmean)**2 for i in range(0,n) for j in range(0,k)]))
    MSBS=SSBS/(n-1)
    SSE=SST-SSBS-SSBM
    MSE=SSE/((n-1)*(k-1))
    return (MSBS-MSE)/(MSBS+(k-1)*MSE)

@njit(float64(float64[:,:]))
def ICC_A1(X):
    """Calculate ICC(A,1) - two-way random effects, absolute agreement model.

    Arguments
    ---------
    X : ndarray, shape (T, N)
        Data where T represents subjects and N represents raters.

    Returns
    -------
    float
        ICC(A,1) value.
    
    References
    ----------
    Liljequist et al. (2019) DOI: 10.1371/journal.pone.0219854
    """  
    n=X.shape[0]
    k=X.shape[1]
    Xmean=np.mean(X)
    S=1/k*np.sum(X, axis=1)
    M=1/n*np.sum(X, axis=0)
    SST=np.sum((X-Xmean)**2)
    SSBS=np.sum(np.array([(S[i]-Xmean)**2 for i in range(0,n) for j in range(0,k)]))
    SSBM=np.sum(np.array([(M[j]-Xmean)**2 for i in range(0,n) for j in range(0,k)]))
    MSBS=SSBS/(n-1)
    MSBM=SSBM/(k-1)
    SSE=SST-SSBS-SSBM
    MSE=SSE/((n-1)*(k-1))
    return (MSBS-MSE)/(MSBS+(k-1)*MSE+k/n*(MSBM-MSE))

@njit(float64[:,:,:](float64[:,:], int16, string, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True)
def numba_rolling_ICC(data, window, kind="ICC(1)", accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate intraclass correlation matrix in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    kind : str, optional
        ICC type: 'ICC(1)', 'ICC(A,1)', or 'ICC(C,1)'. Default 'ICC(1)'.
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
    ndarray, shape (T, N, N)
        ICC matrix at each time point.
    
    References
    ----------
    Liljequist et al. (2019) DOI: 10.1371/journal.pone.0219854
    """    
    if kind == "ICC(1)":
        ICC_func=ICC_1
    elif kind == "ICC(A,1)":
        ICC_func=ICC_A1
    elif kind == "ICC(C,1)":
        ICC_func=ICC_C1
    else:
        raise ValueError("Given type of ICC is unknown. Valid entries are ICC(1), ICC(A,1), ICC(C,1)")
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert window >1 # Otherwise division by zero possible
    idx_list=[[i,j] for i in range(0,N) for j in range(0,N) if i<=j  ] # Same as list(combinations_with_replacement(range(N),2))
    idx_tpl=[[int(x[i]) for x in idx_list] for i in range(0,2)]
    N_idx = len(idx_list)
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N,N), dtype=float)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :,:]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        _ICC=np.zeros((N,N), dtype=float)
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        if w.shape[0]<=1:
            windows[t+window-1,:,:]=np.nan
        else:
            _uniqueRes=[ICC_func(np.column_stack((w[:,i],w[:,j]))) for i,j in idx_list]
            for i in range(N_idx):
                _ICC[idx_tpl[0][i], idx_tpl[1][i]]=_uniqueRes[i]
            windows[t+window-1]=_ICC + np.tril(_ICC.T, -1)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

#endregion
#region Cross-Entropy

# Note: EntropyHub (https://pypi.org/project/EntropyHub/) and py-msentropy 
# (https://github.com/antoine-jamin/py-msentropy) also provide cross-entropy implementations
# but produce slightly different results despite citing the same reference. 
# This implementation uses custom Numba optimization for better performance.
@njit(float64(float64[:], float64[:], int16, float64), parallel=True, fastmath=True)  # Parallel improves speed by ~20x
def XSampEn(u, v, m, r):
    """Calculate cross-sample entropy between two time series."""
    if (len(u) != len(v)):
        raise Exception("Error: length of u different than length of v")
    N = u.shape[0]
    dim=N-m
    A_d=np.empty((dim,dim))
    B_d=np.empty((dim,dim))
    samples = sliding_window_view(np.vstack((u,v)).T, m+1, axis=0)
    for i in prange(0, dim ):
        for j in prange(0, dim ):
            B_d[i,j] = np.max(np.abs(samples[i,0,:-1] - samples[j,1,:-1]))  # Optimization as of March 3, 2024
            A_d[i,j] = np.max(np.abs(samples[i,0,:] - samples[j,1,:])) 
    totA=np.sum(A_d<=r)
    totB=np.sum(B_d<=r)
    if totB == 0 or totA ==0: # Return maximum if no match
        cse = np.log(N-m-1)+np.log(N-m)-np.log(2) #np.nan # Catch if no match was found
    else:
        cse = -np.log((totA/dim) / (totB/dim))/np.log(np.exp(1))
    return cse


@njit(float64(float64[:], float64[:], int16, float64), parallel=True, fastmath=True)
def XApEn(u, v, m, r):
    """Calculate cross-approximate entropy between two time series."""
    N=u.shape[0]
    dim=N-m+1
    samples1 = sliding_window_view(np.vstack((u,v)).T, m, axis=0)
    samples2 = sliding_window_view(np.vstack((u,v)).T, m+1, axis=0)
    C=np.empty((2,dim))
    for i in prange(dim):
        res = np.zeros(2)
        for j in prange(dim):
            if np.max(np.abs(samples1[i,0]-samples1[j,1]))<=r:
                res[0] +=1
                if i<dim-1 and j<dim-1 and np.max(np.abs(samples2[i,0]-samples2[j,1]))<=r:
                    res[1] +=1
        C[0,i]=res[0] / dim 
        C[1,i]=res[1] / (dim-1)
    C=np.where(C==0, 1, C)
    return np.abs((1/(dim) * np.sum(np.log(C[0])))- (1/(dim-1) * np.sum(np.log(C[1]))))

@njit(float64[:,:,:](float64[:,:], int16, int16, float64, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=False)
def numba_rolling_crossSampEn(data, window, m=2, r=0.2, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate cross-sample entropy matrix in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    m : int, optional
        Embedding dimension. Default 2.
    r : float, optional
        Tolerance threshold (fraction of std). Default 0.2.
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
    ndarray, shape (T, N, N)
        Cross-sample entropy matrix at each time point.
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert window >1 # Otherwise division by zero possible
    idx_list=[[i,j] for i in range(0,N) for j in range(0,N) if i<=j  ] # Same as list(combinations_with_replacement(range(N),2))
    idx_tpl=[[int(x[i]) for x in idx_list] for i in range(0,2)]
    N_idx = len(idx_list)
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N,N), dtype=float)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :,:]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        _entropy=np.zeros((N,N), dtype=float)
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        if (t>m+1):
            _uniqueRes=[XSampEn(w[:,i],w[:,j], m=int(m), r= r * np.nanstd(np.column_stack((w[:,int(i)],w[:,int(j)])))) for i,j in idx_list]
        else:
            _uniqueRes=[np.nan for i,j in idx_list]
        for i in range(N_idx):
            _entropy[idx_tpl[0][i], idx_tpl[1][i]]=_uniqueRes[i]
        windows[t+window-1]=_entropy + np.tril(_entropy.T, -1)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

@njit(float64[:,:,:](float64[:,:], int16, int16, float64, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=False)
def numba_rolling_crossApEn(data, window, m=2, r=0.2, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Calculate cross-approximate entropy matrix in rolling windows.
    
    Arguments
    ---------
    data : ndarray, shape (T, N)
        Input time-series data.
    window : int
        Rolling window size.
    m : int, optional
        Embedding dimension. Default 2.
    r : float, optional
        Tolerance threshold (fraction of std). Default 0.2.
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
    ndarray, shape (T, N, N)
        Cross-approximate entropy matrix at each time point.
    """
    data, window = apply_PreProcessing(preProcessing,data,window, preParameters)
    T=data.shape[0]
    N=data.shape[1]
    assert window >1 # Otherwise division by zero possible
    idx_list=[[i,j] for i in range(0,N) for j in range(0,N) if i<=j  ] # Same as list(combinations_with_replacement(range(N),2))
    idx_tpl=[[int(x[i]) for x in idx_list] for i in range(0,2)]
    N_idx=len(idx_list)
    assert T>=0 and N>=0 and window >=0
    windows=np.empty((T, N,N), dtype=float)
    sliding_window =np.swapaxes(
        sliding_window_view(data, window, axis=0),
        1,2).copy()
    windows[:window, :,:]=np.nan
    for t in prange(1, sliding_window.shape[0]):
        _entropy=np.zeros((N,N), dtype=float)
        w = sliding_window[t]
        w= apply_midProcessing(midProcessing, w, midParameters)
        if (t>m+1):
            _uniqueRes=[XApEn(w[:,i],w[:,j], m=int(m), r= r * np.nanstd(np.column_stack((w[:,int(i)],w[:,int(j)])))) for i,j in idx_list]
        else:
            _uniqueRes=[np.nan for i,j in idx_list]
        for i in range(N_idx):
            _entropy[idx_tpl[0][i], idx_tpl[1][i]]=_uniqueRes[i]
        windows[t+window-1]=_entropy + np.tril(_entropy.T, -1)
    parameters = List([float64(accuracy)])
    windows = apply_PostProcessing(postProcessing, windows, parameters)
    return windows

#endregion 
