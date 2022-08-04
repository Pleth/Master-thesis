import numpy as np
from scipy import interpolate
from tqdm.auto import tqdm

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
    assert (y_reference.shape[0] == x_reference.shape[0]), f"y_reference and x_reference size missmatch in first diemension ({y_reference.shape[0]}!={x_reference.shape[0]})"

    f = interpolate.interp1d(x_reference, y_reference, kind=kind)
    return (x_observation, f(x_observation))


def moving_avg(a:np.array, k:int=10):
    """
    Computes the k-point moving average of the array.

    :param numpy.array a: Input array for which the moving average is to be computed.
    :param int k: Optional, number of points to use in computing the moving average. Defaults to 10.
    """
    n = len(a)
    b = np.zeros(n)
    for i in range(n-1):
        if i+k < n:
            b[i] = a[i:i+k].mean()
        else:
            b[i] = a[i:n].mean()
    b[-1] = a[-1]
    return b


def create_synthetic_signal(p79_distances:np.array, p79_laser5:np.array, p79_laser21:np.array,
                            gm_times:np.array, gm_speed:np.array):
    """
    Creates the synthetic accelerometer signal using the p79 profile. Using the lasers at the
    wheels (5,21) the average profile is computed, then smoothed using 5-points moving average.
    A road observation is made using the Green Mobility speed, and finally the quarter-car model
    outputs the simulated accelerations.

    :param np.array p79_distances: numpy array containing the p79 distance mesurements, which are used to timestamp the p79 measurements. Units in meters.
    :param np.array p79_laser5: numpy array containing the measurements from the laser number 5, on the left wheel. Units in mm.
    :param np.array p79_laser21: numpy array containing the measurements from the laser number 21, on the right wheel. Units in mm.
    :param np.array gm_times: numpy array containing the time measurements, which are used to timestamp the Green Mobility speed measurements. Units in seconds. 
    :param np.array gm_speed: numpy array containing the measurements of the Green Mobility speed. Units in km/h.
    :returns: dictionary with keys "times", "synth_acc", "start_time" which correspond to time labels for the generated synthetic accelerations (s), accelerations (in g's, 1g = 9.81 m/s^2), and start time reference (s), because the time labels are shifted to start at 0, so we anotate what that 0 is really.
    :rtype: dict
    """
    ## PROCESS THE GM SIGNALS
    ##########################################
    # sort arrays in incresing time/distance
    idx_gm = np.argsort(gm_times)
    gm_times = gm_times[idx_gm]
    gm_speed = gm_speed[idx_gm]

    # TODO: remove, this can be used to artificially create a bias in the gm speed
    # and check how the resulting misslalignment is.
    gm_speed = gm_speed*1.00

    # remove duplicated timestamps
    idxs = np.argwhere(np.diff(gm_times) != 0.0)
    gm_times = gm_times[idxs][:,0]
    gm_speed = gm_speed[idxs][:,0]

    # smooth speed
    gm_speed = moving_avg(gm_speed)

    # resample at constant sf=250Hz
    new_gm_times = np.arange(gm_times.min(), gm_times.max(), 1/250)
    gm_times, gm_speed = make_observation(y_reference=gm_speed, x_reference=gm_times, x_observation=new_gm_times)

    # establish the time=0s at the first point
    gm_start_time = gm_times.min() # s
    gm_times = gm_times - gm_start_time # s

    # compute GM distances using times and speeds
    gm_speed = gm_speed*(1000/3600) # km/h -> m/s
    gm_distances = np.cumsum(gm_speed[:-1]*np.diff(gm_times)) # m

    # establish the distance=0m at the first point
    min_gm_distances = gm_distances.min()
    gm_distances = gm_distances - min_gm_distances

    ## PROCESS THE P79 SIGNAL
    ##########################################
    # sort arrays in incresing time/distance
    idx_p79 = np.argsort(p79_distances)
    p79_distances = p79_distances[idx_p79]

    # establish the distance=0m at the first point
    min_p79_distance = p79_distances.min()
    p79_distances = p79_distances - min_p79_distance
    p79_laser21 = p79_laser21[idx_p79]
    p79_laser5 = p79_laser5[idx_p79]
    
    # compute raw p79 profile and smooth with moving average
    p79_profile = (p79_laser5 + p79_laser21)/2 # mm
    p79_profile = moving_avg(p79_profile, 5)/1e3 # mm -> m

    ## CREATE THE ROAD OBSERVATION AS GM CAR WOULD HAVE SEEN IT DRIVING ITS OWN SPEED   
    # in order to interpolate, we need to grab the common segment of both sequences
    idxs_gm = np.argwhere((gm_distances < p79_distances.max()) & (gm_distances > p79_distances.min()))
    gm_distances = gm_distances[idxs_gm]
    gm_times = gm_times[idxs_gm]
    
    
    # interpolate to get the profile GM would observe, NOTE: the resulting profile is then sampled at 250 Hz
    gm_distances, gm_profile = make_observation(y_reference=p79_profile, x_reference=p79_distances, x_observation=gm_distances)
    
    # finally compute the synthetic accelerations
    print("Generating Synthetic Profile ...")
    synth_acc, _, _ = Car().drive(gm_times, gm_profile)

    return {"times":gm_times, "synth_acc":synth_acc, "start_time":gm_start_time,"p79_distances": gm_distances+min_p79_distance}


class Car:
    """
    Class that manages the reactions of the car as it drives on a road profile.
    
    :param str car_model: string label of the car model (defaults None).
    """
    def __init__(self, car_model:str=None):

        # if car model specify load its parameters
        if car_model:
            self._load_car_params(car_model=car_model)

            self.K1 = self.kt/self.ms
            self.K2 = self.ks/self.ms
            self.C = self.cs/self.ms
            self.U = self.mu/self.ms

        # if car model not defined then use the defaults from the paper
        else:
            self.K1 = 318.61
            self.K2 = 132.32
            self.C = 5.68
            self.U = 0.093
    
    
    def _load_car_params(self, car_model:str):
        """
        Load the parameters corresponding to the selected model of car
        :param car_model: string label of the car model.
        """
        # car parameters, (tuned for handling (Allison 2008))
        if car_model == "allison":
            self.ms = 325       # 1/4 sprung mass (kg)
            self.mu =  65       # 1/4 unsprung mass (kg)
            self.ks = 505.33   # suspension stiffness
            self.kt = 232.5e3  # tire stiffness (N/m)
            self.cs = 1897.9   # suspension damping coefficient

        # Quarter-car Renault Megane Coupe (Santos 2010)
        elif car_model == "renault":
            self.ms = 315       # 1/4 sprung mass (kg)
            self.cs = 603.9     # suspension damping coefficient
            self.ks = 29500     # suspension stiffness
            self.kt = 210000    # tire stiffness
            self.mu = 37        # 1/4 unsprung mass (kg)
        
        # Quarter-car BMW 530i (van der Sande 2013)
        elif car_model == "bmw":
            self.ms = 395.3     # 1/4 sprung mass (kg)
            self.cs = 1450      # suspension damping coefficient
            self.ks = 30.01e3   # suspension stiffness
            self.kt = 3.4e5     # tire stiffness
            self.mu = 48.3      # 1/4 unsprung mass (kg)
        
        # manage exception
        else:
            raise Exception(f"Car model not supported. Try 'allison', 'renault' or 'bmw'. Your input:{car_model}")


    def _unsprung_mass_displacement(self, dt:float, Zp:float):
        """
        Computes the update on the displacement of the unsprung mass (lower mass) Zu
        :param dt: float time differential
        :param Zp: float value heigth of the road (m)
        :return: returns Zu+, the updated value of the displacement of the unsprung mass.
        """
        idx = self.idx
        term1 = (dt*self.C+2)*((dt**2)*self.K1*(Zp-self.Zu[idx-1]) - self.U*(self.Zu[idx-2] - 2*self.Zu[idx-1]) + 2*self.Zs[idx-1] - self.Zs[idx-2])
        term2 = 2*(dt**2)*self.K2*(self.Zs[idx-1]-self.Zu[idx-1])
        term3 = dt*self.C*(self.Zu[idx-2]-self.Zs[idx-2]) +2*self.Zs[idx-2] - 4*self.Zs[idx-1]
        return (term1+term2+term3)/(dt*self.C*(1+self.U)+2*self.U)


    def _sprung_mass_displacement(self, dt:float, Zp:float, Zu:float):
        """
        Computes the update on the displacement of the sprung mass (upper mass) Zs
        :param dt: float time differential
        :param Zp: float value heigth of the road (m)
        :param Zu: float the updated value of the displacement of the unsprung mass (its Zu+)
        :return: returns Zs+, the updated value of the displacement of the sprung mass.
        """
        idx = self.idx
        return (dt**2)*self.K1*(Zp-self.Zu[idx-1])-self.U*(Zu-2*self.Zu[idx-1] + self.Zu[idx-2]) +2*self.Zs[idx-1] - self.Zs[idx-2]


    def _compute_z_acceleration(self, dt:float, Zs:float):
        """
        Computes the update on the acceleration of the displacement of the sprung mass
        :param dt: float time differential
        :param Zs: float the updated value of the displacement of the sprung mass (its Zs+)
        :return: returns Zs_dot_dot+, the updated float value of the acceleration in the displacement of the sprung mass.
        """
        idx = self.idx
        return (Zs-2*self.Zs[idx-1]+self.Zs[idx-2])/(dt**2)


    def _compute_delta_t(self, time_slice:list):
        """
        Computes the time differential delta_t given a time slice by averaging the differences in timesteps.
        :param time_slice: list(float) containing values of the timesteps (one before, current, one after).
        :return: float value of the time differential delta_t
        :rtype: float
        """
        try:
            return (time_slice[2]-time_slice[0])/2
        except:
            return self.prev_dt


    def drive(self, time:list, Zp:list):
        """
        Given a list of timesteps and heights of the road corresponding to those timesteps
        it simulates the car driving over the profile and saves the acceleration of the displacement of the 
        sprung mass (upper mass).

        :param list time: List containing the times (in seconds) at which the different measurements of the profile Zp were taken.
        :param list Zp: List containing the values of the  measurements of the profile.
        :returns: tuple containg the array of synthetic accelerations of the sprung mass (in g's, 1g=9.81 m/s²), displacements of the sprung mass and displacements of the unsprung mass.
        :rtype: tuple
        """
        # sanity check: check time and profile measuremets have the same length
        assert len(time) == len(Zp), f"time and profile (Zp) vectors must have the same length. {len(time)}!={len(Zp)}"

        # initialize reactions (assume they start with system at rest)
        n_times = len(time)
        self.Zu, self.Zs, self.Zs_dot_dot = np.zeros(n_times+2), np.zeros(n_times+2), np.zeros(n_times+2)

        for i in tqdm(range(n_times)):
            self.idx = i+2
            
            # in case we havent seen more than 2 timesteps get the differential time 
            # from a future slice
            if i < 1:
                dt = self._compute_delta_t(time_slice=time[0:3])
            else:
                dt = self._compute_delta_t(time_slice=time[(i-1):(i+2)])
            
            self.prev_dt = dt

            # compute the reactions
            Zu = self._unsprung_mass_displacement(dt=dt, Zp=Zp[i])
            Zs = self._sprung_mass_displacement(dt=dt, Zp=Zp[i], Zu=Zu)
            Zs_dot_dot = self._compute_z_acceleration(dt=dt, Zs=Zs)
            
            # update the array
            self.Zu[i+2], self.Zs[i+2], self.Zs_dot_dot[i+2] = Zu, Zs, Zs_dot_dot
        
        # convert accelerations from m/s² to g units (1g = 9.81 m/s²)
        self.Zs_dot_dot = self.Zs_dot_dot/9.81
        
        # return only from the second term onwards (the 2 first were there for initialization)
        return (self.Zs_dot_dot[2:], self.Zs[2:], self.Zu[2:])