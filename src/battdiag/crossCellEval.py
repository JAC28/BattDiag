#region Import of Packages
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from numba import jit,njit, float64, int16, boolean, prange 
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
    result = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        result[i] = np.std(X[:,i])
    return result

## Numba implementation of zScore over column axes
@njit(float64[:,:,:](float64[:,:,:], int16), parallel=True, fastmath=True, cache=True)
def numba_3D_zScore(data,accuracy=8):
    T,N,N = data.shape
    zScore=np.empty_like(data)
    for t in prange(T):
        zScore[t,:,:] = np.divide(np.subtract(data[t,:,:],np.nanmean(data[t,:,:])), np.nanstd(data[t,:,:]))
    return np.around(zScore, accuracy)

## Function downsamples data by fraction and returns updated window
@njit(float64[:,:](float64[:,:], int16, string), parallel=True, fastmath=True, cache=True)
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
                raise ValueError("Given reduction method is invalid.")
    return result
#endregion

#region Pre-, Data- and Postprocessing
## Pre-processing
@njit(Tuple((float64[:,:], int16))(string, float64[:,:], int16, ListType(float64)), fastmath=True, parallel=False, cache=True)
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
@njit(float64[:,:](string, float64[:,:],ListType(float64)), fastmath=True, cache=True) # Parallel verlangsamt Ausführung 
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
@njit(float64[:,:,:](string, float64[:,:,:], ListType(float64)), fastmath=True, cache=True)
def apply_PostProcessing(postProcessing, data, parameters):
    if postProcessing == "None":
        accuracy = int16(parameters[0])
        return np.around(data, accuracy)
    elif postProcessing == "zScore":
        accuracy = int16(parameters[0])
        return numba_3D_zScore(data, accuracy)
    else:
        raise ValueError("Given postprocessing method is invalid.")
#endregion

#region Correlation
## Pearson Correlation
@njit(float64[:,:,:](float64[:,:], int16, int16, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True) 
def numba_rolling_PearCorr(data, window, accuracy=8, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Cross-column correlation with rolling window. 
    Implementation does not use any optimization regarding rolling window
    but is quite fast due to numba-usage. 

    Args:
        data (ndarray): 2D array containing the time-series data. Expected shape is 
                        (T,N), where T and N represents the total time and number 
                        of parallel channels, respectively. 
        window (int): Lenght of observer window.
        accuracy (int, optional): Controll over the significant places of the result.
                                  Defaults to 8.
        rect (boolean, optional): Flag to control whether the data is superimposed by a rectangle
                                  signal. Default to False.
        A (float, optional): Amplitude of rectangle signal relative to median of STD. Default is 0.1.
        P (int, optional): Number of periods of rectangle signal added to the signal. Signal is distributed
                           over the given sample size. Defaults to 2.

    Returns:
        ndarray: 3D array containing the Cross-Pearson-Correlation values at each point in time.
                 Shape of result is (T,N,N), where T and N represents the total time and number
                 of parallel channels, respectively.
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
@njit(float64(float64[:,:]))
def ICC_1(X):
    """Numba implementation of default ICC ICC(1) as introduced by Fisher.
    See Liljequist.2019 for reference.

    Args:
        X (ndarray): Data for which the ICC is calculated. Expected shape is
                     (T,N), where T and N represent the "subject" and the "rater",
                     respectively.

    Returns:
        ndarray: ICC(1) value of the given data.
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
    """Numba implementation of ICC ICC(C,1).
    See Liljequist.2019 for reference.

    Args:
        X (ndarray): Data for which the ICC is calculated. Expected shape is
                     (T,N), where T and N represent the "subject" and the "rater",
                     respectively.

    Returns:
        ndarray: ICC(C,1) value of the given data.
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
    """Numba implementation of ICC ICC(A,1).
    See Liljequist.2019 for reference.

    Args:
        X (ndarray): Data for which the ICC is calculated. Expected shape is
                     (T,N), where T and N represent the "subject" and the "rater",
                     respectively.

    Returns:
        ndarray: ICC(A,1) value of the given data.
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

@njit(float64[:,:,:](float64[:,:], int16, int16, string, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=True)
def numba_rolling_ICC(data, window, accuracy=8, kind="ICC(1)", preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):
    """Cross-column ICC calculation of various kinds with rolling window. 
    Implementation does not use any optimization regarding rolling window
    but is quite fast due to numba-usage.
    Optional Data-preprocessing by addition of rectangle signal. 

    Args:
        data (ndarray): Data for which the ICC is calculated. Expected shape is
                        (T,N), where T and N represent the "subject" and the "rater",
                        respectively.
        window (int): Lenght of observer window.
        accuracy (int, optional): Controll over the significant places of the result.
                                  Defaults to 8.
        kind (str, optional): Defines the kind of ICC [ICC(1), ICC(A,1), ICC(C,1)] calculated. Defaults to "ICC(1)".
        rect (bool, optional): Flag to control whether the data is superimposed by a rectangle
                                  signal. Defaults to False.
        A (float, optional): Amplitude of rectangle signal relative to median of STD. Defaults to 0.1.
        P (int, optional): Number of periods of rectangle signal added to the signal. Signal is distributed
                           over the given sample size. Defaults to 2.

    Raises:
        ValueError: If the provided kind is not implemented. 

    Returns:
        ndarray: 3D array containing the Intraclass-Correlation values at each point in time.
                 Shape of result is (T,N,N), where T and N represents the total time and number
                 of parallel channels, respectively.
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
@njit(float64(float64[:], float64[:], int16, float64), parallel=True, fastmath=True) # Parallel inmproves speed by 20x
def XSampEn(u, v, m, r):
    if (len(u) != len(v)):
        raise Exception("Error : lenght of u different than lenght of v")
    N = u.shape[0]
    dim=N-m
    A_d=np.empty((dim,dim))
    B_d=np.empty((dim,dim))
    samples = sliding_window_view(np.vstack((u,v)).T, m+1, axis=0)
    for i in prange(0, dim ):
        for j in prange(0, dim ):
            B_d[i,j] = np.max(np.abs(samples[i,0,:-1] - samples[j,1,:-1]))  #Optimierung Stand 3.3.24
            A_d[i,j] = np.max(np.abs(samples[i,0,:] - samples[j,1,:])) 
    totA=np.sum(A_d<=r)
    totB=np.sum(B_d<=r)
    if totB == 0 or totA ==0:
        cse = np.log(N-m-1)+np.log(N-m)-np.log(2) #np.nan # Catch if no match was found
    else:
        cse = -np.log((totA/dim) / (totB/dim))/np.log(np.exp(1))
    return cse


@njit(float64(float64[:], float64[:], int16, float64), parallel=True, fastmath=True)
def XApEn(u, v, m, r):
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

@njit(float64[:,:,:](float64[:,:], int16, int16, int16, float64, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=False)
def numba_rolling_crossSampEn(data, window, accuracy=8, m=2 , r=0.2, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):     
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

@njit(float64[:,:,:](float64[:,:], int16, int16, int16, float64, string, ListType(float64), string, ListType(float64), string, ListType(float64)), parallel=False)
def numba_rolling_crossApEn(data, window, accuracy=8, m=2 , r=0.2, preProcessing="None", preParameters=List([10.0]),  midProcessing = "None", midParameters=List([0.1,2.0]), postProcessing="None", postParameters=List([1.0])):     
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
