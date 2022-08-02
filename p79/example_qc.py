import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import interpolate
import math

from car import Car

def make_observation(y_reference:np.array, x_reference:np.array, x_observation:np.array, kind:str='linear'):
    """
    Given a signal (y_reference, x_reference) return an observation of it at the x_observations points.
    Interpolates a signal on the given points.

    :param numpy.array y_reference: signal values of the signal to be observed.
    :param numpy.array x_reference: time values/ points at which the signal is observed initially.
    :param numpy.array x_observation: time values/ points at which the signal is  wants to be observed.
    :param str kind: Optional, Type of method to use for interpolation (e.g 'linear', 'cubic'). Defaults to 'linear'
    :returns: Tuple of x_observation and Interpolated values of the reference signal at the x_observation values.
    :rtype: tuple
    """
    assert (y_reference.shape[0] == x_reference.shape[0]), f"y_reference and x_reference size missmatch in first dimension ({y_reference.shape[0]}!={x_reference.shape[0]})"

    f = interpolate.interp1d(x_reference, y_reference, kind=kind)
    return (x_observation, f(x_observation))
    
if __name__ == "__main__":

    file_p79 = "data/p79_data.csv"
    file_gm = "data/green_mob_data.csv"


    df = pd.read_csv(file_gm) # speed_gm,times

    plt.plot(df["times"], df["speed_gm"])
    plt.show()

    plt.close()


    
    df = pd.read_csv(file_p79) #distances,laser5,laser21

    plt.plot(df["distances"], df["laser5"])
    plt.plot(df["distances"], df["laser21"])
    plt.show()


    #mycar = Car()
    #mycar.drive(time=[], Zp=[])