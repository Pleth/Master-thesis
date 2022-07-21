import numpy as np
from tqdm.auto import tqdm

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