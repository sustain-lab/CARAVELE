from datetime import datetime, timedelta
import glob
import numpy as np
from scipy.signal import detrend
from toa5 import read_irgason, read_udm, read_wavewire
import xarray as xr

DATAPATH = 'data'
RUN_SECONDS = 900
FREQUENCY = 20
NUM_RUNS = 5

start_times = [
    datetime(2021, 4, 20, 17, 0),
    datetime(2021, 4, 20, 17, 25),
    datetime(2021, 4, 20, 17, 50),
    datetime(2021, 4, 20, 18, 15),
    datetime(2021, 4, 20, 18, 35)
]

end_times = [start + timedelta(seconds=RUN_SECONDS)
    for start in start_times]

seconds = np.arange(0, RUN_SECONDS + 1 / FREQUENCY, 1 / FREQUENCY)

# IRGASON
files = glob.glob(DATAPATH + '/TOA5_SUSTAIN_Wind.FAST*.dat')
files.sort()
irg_time, _, irg = read_irgason(files, valid_flag=11)

# Wave wires
files = glob.glob(DATAPATH + '/TOA5_OSSwave*.dat')
files.sort()
wavewire_time, wavewire_data = read_wavewire(files)

# UDM
files = glob.glob(DATAPATH + '/TOA5_SUSTAIN_ELEV*.dat')
files.sort()
udm_time, u1, u2, u3, u4, u5, u6 = read_udm(files)

u = np.zeros((NUM_RUNS, seconds.size))
v = np.zeros((NUM_RUNS, seconds.size))
w = np.zeros((NUM_RUNS, seconds.size))
T = np.zeros((NUM_RUNS, seconds.size))
eta_udm = np.zeros((NUM_RUNS, seconds.size))
eta_wire = np.zeros((NUM_RUNS, seconds.size))

for n in range(NUM_RUNS):
    start_time = start_times[n]
    end_time = end_times[n]

    mask = (irg_time >= start_time) & (irg_time <= end_time)
    ttime = irg_time[mask]
    irg_seconds = np.array([(t - start_time).total_seconds() for t in ttime])
    u[n,:] = np.interp(seconds, irg_seconds, irg['u'][mask])
    v[n,:] = np.interp(seconds, irg_seconds, irg['v'][mask])
    w[n,:] = np.interp(seconds, irg_seconds, irg['w'][mask])
    T[n,:] = np.interp(seconds, irg_seconds, irg['T'][mask])

    mask = (wavewire_time >= start_time) & (wavewire_time <= end_time)
    ttime = wavewire_time[mask]
    wavewire_seconds = np.array([(t - start_time).total_seconds() for t in ttime])
    eta_wire[n,:] = detrend(np.interp(seconds, wavewire_seconds, wavewire_data['w4'][mask]))

    mask = (udm_time >= start_time) & (udm_time <= end_time)
    ttime = udm_time[mask]
    udm_seconds = np.array([(t - start_time).total_seconds() for t in ttime])
    eta_udm[n,:] = np.interp(seconds, udm_seconds, u4[mask])

# remove drop outs in UDM
for npass in range(20):
    for n in range(NUM_RUNS):
        for i in range(1, seconds.size - 1):
            if eta_udm[n,i] > 1.2 + 0.01 * n:
                eta_udm[n,i] = 0.5 * (eta_udm[n,i-1] + eta_udm[n,i+1])

for n in range(NUM_RUNS):
    eta_udm[n,:] = - detrend(eta_udm[n,:])

# Create dataset
ds = xr.Dataset(
    {
        'fan': ('runs', range(15, 40, 5)), 
        'u': (['runs', 'time'], u), 
        'v': (['runs', 'time'], v), 
        'w': (['runs', 'time'], w), 
        'T': (['runs', 'time'], T),
        'eta_wire': (['runs', 'time'], eta_wire),
        'eta_udm': (['runs', 'time'], eta_udm),
    },
    coords = {
        'runs': range(1, 6),
        'time': seconds
    }
)

# Add metadata
ds['time'].attrs['name'] = 'Time since start of experiment'
ds['time'].attrs['units'] = 's'
ds['fan'].attrs['name'] = 'Fan speed'
ds['fan'].attrs['units'] = 'Hz'
ds['u'].attrs['name'] = 'Along-tank velocity from IRGASON'
ds['u'].attrs['units'] = 'm/s'
ds['u'].attrs['fetch'] = 9.55
ds['v'].attrs['name'] = 'Cross-tank velocity from IRGASON'
ds['v'].attrs['units'] = 'm/s'
ds['v'].attrs['fetch'] = 9.55
ds['w'].attrs['name'] = 'Vertical velocity from IRGASON'
ds['w'].attrs['units'] = 'm/s'
ds['w'].attrs['fetch'] = 9.55
ds['T'].attrs['name'] = 'Air temperature from IRGASON'
ds['T'].attrs['units'] = 'K'
ds['T'].attrs['fetch'] = 9.55
ds['eta_wire'].attrs['name'] = 'Water elevation from wave wire'
ds['eta_wire'].attrs['units'] = 'm'
ds['eta_udm'].attrs['name'] = 'Water elevation from UDM'
ds['eta_udm'].attrs['units'] = 'm'

ds.attrs['experiment_name'] = 'CARAVALE'
ds.attrs['experiment_time'] = start_times[0].strftime('%Y-%m-%d_%H:%M:%S')
ds.attrs['water_type'] = 'seawater'
ds.attrs['initial_water_depth'] = 0.81
ds.attrs['institution'] = 'University of Miami'
ds.attrs['facility'] = 'SUSTAIN Laboratory'
ds.attrs['tank'] = 'SUSTAIN'
ds.attrs['contact'] = 'Milan Curcic <mcurcic@miami.edu>'

ds.to_netcdf('caravale_sustain.nc', 'w', 'NETCDF4')
