import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from math import radians, cos, sin, asin, sqrt, ceil, floor
import numpy as np

def preprocess_gm(passage, acc_sig = 'acc.xyz', signals = ['obd.spd', 'gps']):
    """Extract the signals and interpolate to match sampling time
    of accelerometer data.  

    Args:
        passage (h5py.Group): Group in HDF5 file containing the relevant sensors
        for the passage. 
        acc_sig (str, optional): Name of the accelerometer data that we want to
        interpolate the remaining signals to. Defaults to 'acc.xyz'.
        signals (list, optional): List of strings containing the names of the signals
        to interpolate. Defaults to ['obd.spd', 'gps'].

    Returns:
        pd.DataFrame: pandas dataframe containing all of the interpolated signals 
    """

    # extract data and interpolate
    acc = passage[acc_sig]
    if 'reorientation' in passage.parent.attrs.keys():
        x = passage.parent.attrs['reorientation']
        acc_xyz = reorient_axis(x, acc[:,1:])
        acc = np.concatenate((acc[:,0][:, np.newaxis], acc_xyz), axis = 1)
    collect_sig = pd.DataFrame(acc, columns = passage[acc_sig].attrs['chNames'])
    if np.mean(collect_sig['acc_z'])<-0.7:
        collect_sig['acc_z'] = -collect_sig['acc_z']
    
    for sig in signals:
        sensor = pd.DataFrame(passage[sig], columns = passage[sig].attrs['chNames'])
        collect_sig = interpolate_sensor(collect_sig, sensor, passage[sig].attrs['chNames'][1:])
        if sig == 'obd.spd':
            # get odometer
            collect_sig['odometer'] = (collect_sig['TS_or_Distance'].diff()*collect_sig['spd']/3.6).cumsum()

    return collect_sig


def reorient_axis(x, acc):
    alpha = x[0]
    beta = x[1]
    gamma = x[2]
    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gamma)
    Rza = np.array([[ca, -sa,  0], [sa,  ca,    0],   [0,  0,   1]])
    Ryb = np.array([[cb,   0, sb],  [0,   1,    0], [-sb,  0,  cb]])
    Rxg = np.array([[1,    0 , 0],  [0,  cg,  -sg],   [0, sg,  cg]])
    
    acc = acc.T
    T  = Rza@Ryb@Rxg
    at = np.zeros((3, acc.shape[1]))
    for i in range(acc.shape[1]):
        at[:, i] = T@acc[:, i]
    return at.T


def interpolate_signal(times, signal, signal_times, kind = 'linear', bounds_error = True):
    """Interpolate signal using linear interpolation
    and evaluate new times. 

    Args:
        times (list or array): times at which to 
        evaluate interpolation. 
        signal (list or array): signal to interpolate
        signal_times (list or array): original 
        sample times from signal. 

    Returns:
        np.NdArray: interpolated signal
    """
    interpolater = interp1d(signal_times, signal, kind = kind, bounds_error=bounds_error)
    interp_signal = interpolater(times)

    return interp_signal

    

def find_min_gps(drd_lat, drd_lon, gm_lat, gm_lon):
    """Find the closest gps points between drd_lat, drd_lon
    and gm_lat, gm_lon

    Args:
        drd_lat (_type_): _description_
        drd_lon (_type_): _description_
        gm_lat (_type_): _description_
        gm_lon (_type_): _description_

    Returns:
        _type_: _description_
    """
    if np.isscalar(gm_lat):
        gm_lat = [gm_lat]
        gm_lon = [gm_lon]
    if np.isscalar(drd_lat):
        drd_lat = [drd_lat]
        drd_lon = [drd_lon]
        
    dist = np.zeros(len(drd_lat))
    dist_idx = np.zeros(len(drd_lat))
    for i, (lat,lon) in enumerate(zip(drd_lat, drd_lon)):    
        temp_dist = np.zeros(len(gm_lat))
        for j, (glat, glon) in enumerate(zip(gm_lat, gm_lon)):
            temp_dist[j] = haversine_distance(lat, glat, lon, glon)
        dist[i] = np.min(temp_dist)
        dist_idx[i] = np.argmin(temp_dist)
    
    drd_idx = int(np.argmin(dist))
    gm_idx = int(dist_idx[drd_idx])

    return drd_idx, gm_idx, dist[drd_idx]



def interpolate_sensor(orig_sig, new_sig, cols, sample_col = 'TS_or_Distance', bounds_error = True):
    """Function for assigning odometer
    to all timepoints of sensor. Timestamps should
    be given in unix timestamps (s).

    Args:
        sensor (pd.DataFrame): DataFrame with sensor data. Must contain 
        column TS_or_Distance with timepoints for each measurement. 
        Remaining columns are not used. 
        odometer (pd.DataFrame): DataFrame with odometer data. Must contain
        column TS_or_Distance with timepoints for each odometer point. 
        odo (str): Name of column in odometer containing odometer.

    Returns:
        pd.DataFrame: Original sensor DataFrame with added columns 
        vel_int containing interpolated odometer signal. 
    """
    if orig_sig[sample_col].iloc[0] < new_sig[sample_col].iloc[0]:
        orig_sig = orig_sig[orig_sig[sample_col]>= new_sig[sample_col].iloc[0]]
    if orig_sig[sample_col].iloc[-1] > new_sig[sample_col].iloc[-1]:
        orig_sig = orig_sig[orig_sig[sample_col]<= new_sig[sample_col].iloc[-1]]
    
    for col in cols:
        orig_sig[col] = interpolate_signal(orig_sig[sample_col].values, new_sig[col].values, new_sig[sample_col].values, bounds_error = bounds_error)

    return orig_sig



def haversine_distance(lat1, lat2, lon1, lon2, in_meters = True):
     
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
      
    # calculate the result
    res = c*r
    if in_meters :
        return 1000*res
    else:
        return res
