import h5py
import pandas as pd
from functions import *

hdf5file = h5py.File('alligned_data/CPH1_VH.hdf5', 'r')
passage = hdf5file.attrs['GM_full_passes']
aligned_passes = hdf5file.attrs['aligned_passes']

passagefile = hdf5file[aligned_passes[0]]
gps = pd.DataFrame(passagefile['gps'], columns = passagefile['gps'].attrs['chNames'])
aligned_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])

acc = pd.DataFrame(passagefile['acc.xyz'], columns = passagefile['acc.xyz'].attrs['chNames'])

aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])
aran_cracks = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Cracks'], columns = hdf5file['aran/trip_1/pass_1']['Cracks'].attrs['chNames'])
aran_potholes = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Pothole'], columns = hdf5file['aran/trip_1/pass_1']['Pothole'].attrs['chNames'])

DI = pd.DataFrame(calc_DI(aran_alligator,aran_cracks,aran_potholes))
DI.columns=['DI']
gt = pd.concat([aran_location,DI],axis=1)

print(acc)
print(gt)