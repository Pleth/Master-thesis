import h5py
import pandas as pd

hdf5file = h5py.File('/Users/theb/Documents/LiRA/data/CPH1_VH.hdf5', 'r')
passage = hdf5file.attrs['GM_full_passes']
aligned_passes = hdf5file.attrs['aligned_passes']

passagefile = hdf5file[aligned_passes[0]]
gps = pd.DataFrame(passagefile['gps'], columns = passagefile['gps'].attrs['chNames'])
aligned_gps = pd.DataFrame(passagefile['aligned_gps'], columns = passagefile['aligned_gps'].attrs['chNames'])

acc = pd.DataFrame(passagefile['acc.xyz'], columns = passagefile['acc.xyz'].attrs['chNames'])

aran_location = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Location'], columns = hdf5file['aran/trip_1/pass_1']['Location'].attrs['chNames'])
aran_alligator = pd.DataFrame(hdf5file['aran/trip_1/pass_1']['Allig'], columns = hdf5file['aran/trip_1/pass_1']['Allig'].attrs['chNames'])