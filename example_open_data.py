# %%
import h5py
import numpy as np
import pandas as pd
from functions import *
import matplotlib.pyplot as plt
from scipy import signal
import scaleogram as scg

hdf5file = h5py.File('aligned_data/CPH1_VH.hdf5', 'r')
passage = hdf5file.attrs['GM_full_passes']
aligned_passes = hdf5file.attrs['aligned_passes']

passagefile = hdf5file[aligned_passes[0]]
gps = pd.DataFrame(passagefile['gps'], columns = passagefile['gps'].attrs['chNames'])
aligned_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])

acc = pd.DataFrame(passagefile['acc.xyz'], columns = passagefile['acc.xyz'].attrs['chNames'])
acc_fs_50 = pd.DataFrame(passagefile['acc_fs_50'], columns = passagefile['acc_fs_50'].attrs['chNames'])
f_dist = pd.DataFrame(passagefile['f_dist'], columns = passagefile['f_dist'].attrs['chNames'])

aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

DI = pd.DataFrame(calc_DI(aran_alligator,aran_cracks,aran_potholes))
DI.columns=['DI']
gt = pd.concat([aran_location,DI],axis=1)

plt.scatter(x=gt['LongitudeFrom'], y=gt['LatitudeFrom'],s=gt['DI']**2,c="red")#c=gt['DI'],cmap='gray')
plt.scatter(x=aligned_gps['lon'], y=aligned_gps['lat'],s=1,c="blue")
plt.show()
# %%
#dist = np.asarray(rm_aligned(aligned_gps,gt))
#plt.plot(dist)
#plt.show()
#max = np.argpartition(dist,-11)[-11:]

start = 0#1800#1400
end = 1600#2200#3000

scales = scg.periods2scales(np.arange(1,60))

plt.plot(acc_fs_50['acc_z'][start:end])
plt.xlabel("time step")
plt.ylabel("acc_fs_50_z")
scg.cws(acc_fs_50['acc_z'][start:end]-acc_fs_50['acc_z'][start:end].mean(),yscale='log')
plt.show()

plt.plot(acc_fs_50['acc_z'][1800:2200])
plt.xlabel("time step")
plt.ylabel("acc_fs_50_z")
scg.cws(acc_fs_50['acc_z'][1800:2200]-acc_fs_50['acc_z'][1800:2200].mean(),yscale='log')
plt.show()

plt.plot(acc_fs_50['acc_z'][1400:3000])
plt.xlabel("time step")
plt.ylabel("acc_fs_50_z")
scg.cws(acc_fs_50['acc_z'][1400:3000]-acc_fs_50['acc_z'][1400:3000].mean(),yscale='log')
plt.show()

# %%
