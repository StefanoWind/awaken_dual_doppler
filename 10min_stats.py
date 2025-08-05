"""
Calculate 10-min stats from virtual tower and SCADA data
"""

import pandas as pd
import xarray as xr
import sys
import numpy as np
import os
import yaml
import glob
from scipy.stats import binned_statistic

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2023-08-03'#start date
    edate='2023-08-04'#end date
    path_config='configs/config_scada.yaml' #config path
    turbine='G2' #selected turbine
else:
    sdate=sys.argv[1]
    edate=sys.argv[2]
    path_config=sys.argv[3]
    turbine=sys.argv[4]

source_lidar='awaken/sa5.sa1.lidar.vt.c0/*nc' #source of lidar data
source_scada='awaken/kp.turbine.z02.00/*csv' #source of scada data
dtime=600#[s] bin size in time

#%% Initialization
#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#read lidar data
files_lidar=glob.glob(os.path.join(config['path_data'],source_lidar))
print(f'Reading {len(files_lidar)} lidar files',flush=True)
lidar=xr.open_mfdataset(files_lidar)
tnum_lid=(lidar.time.values-np.datetime64('1970-01-01T00:00:00'))/ np.timedelta64(1, "s")

#read scada data
files_scada=glob.glob(os.path.join(config['path_data'],source_scada))
time = np.array([],dtype='datetime64')
power = []
ws = []
yaw = []
for f in files_scada:
    scada = pd.read_csv(f)
    time=np.append(time, pd.to_datetime(scada.iloc[:, 0].values).to_numpy(dtype='datetime64[ns]'))
    power=np.append(power, np.array(scada[f"PKGP1HIST01.OKWF001_KP_Turbine{turbine}.ActivePower"]))
    ws=np.append(ws, np.array(scada[f"PKGP1HIST01.OKWF001_KP_Turbine{turbine}.WindSpeed"]))
    yaw=np.append(yaw, np.array(scada[f"PKGP1HIST01.OKWF001_KP_Turbine{turbine}.Nacelle_Position"]))
    print(f'Reading {f}',flush=True)
    
#create scada xarray dataset
scada=xr.Dataset()
scada['power']=xr.DataArray(power,coords={'time':time})
scada['ws']=xr.DataArray(ws,coords={'time':time})
scada['yaw']=xr.DataArray(yaw,coords={'time':time})
tnum_sca=(scada.time.values-np.datetime64('1970-01-01T00:00:00'))/ np.timedelta64(1, "s")

#%% Main
    
#define time vector (datetime64)
time_bins = np.arange(np.datetime64(sdate+'T00:00:00'),
                      np.datetime64(edate+'T00:00:00')+np.timedelta64(1, "D")+np.timedelta64(dtime, "s")/2,
                      np.timedelta64(dtime, "s"))
time_avg=time_bins[:-1]+np.timedelta64(dtime, "s")/2

#convert timevector to timestamp
tnum_bins=(time_bins-np.datetime64('1970-01-01T00:00:00'))/ np.timedelta64(1, "s")
    
#lidar stats#############################################
       
#exclude and count nans
bin_mask = ~np.isnan(lidar.time.values) & ~np.isnan(lidar.ws.values) & ~np.isnan(lidar.wd.values)
data_avail_lid = binned_statistic(tnum_lid, bin_mask, statistic="mean", bins=time_bins)[0]

#mean wind spped
ws_lid_avg = binned_statistic(tnum_lid[bin_mask], lidar.ws.values[bin_mask], statistic="mean", bins=time_bins)[0]

#stdev of wind speed
ws_lid_std = binned_statistic(tnum_lid[bin_mask], lidar.ws.values[bin_mask], statistic="std", bins=time_bins)[0]

#mean wind direction
cos = binned_statistic(tnum_lid[bin_mask], np.cos(np.radians(lidar.wd.values[bin_mask])), statistic="mean", bins=time_bins)[0]
sin = binned_statistic(tnum_lid[bin_mask], np.sin(np.radians(lidar.wd.values[bin_mask])), statistic="mean", bins=time_bins)[0]
wd_lid_avg=np.degrees(np.arctan2(sin,cos))%360

#scada stats##################################################

#exclude and count nans
bin_mask = ~np.isnan(scada.time.values) & ~np.isnan(scada.ws.values) & ~np.isnan(scada.yaw.values)& ~np.isnan(scada.power.values)
data_avail_sca = binned_statistic(tnum_sca, bin_mask, statistic="mean", bins=time_bins)[0]

#mean of power
power_sca_avg = binned_statistic(tnum_sca[bin_mask], scada.power.values[bin_mask], statistic="mean", bins=time_bins)[0]

#stdev of power
power_sca_std = binned_statistic(tnum_sca[bin_mask], scada.power.values[bin_mask], statistic="std", bins=time_bins)[0]

#mean wind speed
ws_sca_avg = binned_statistic(tnum_sca[bin_mask], scada.ws.values[bin_mask], statistic="mean", bins=time_bins)[0]

#stdev of wind speed
ws_sca_std = binned_statistic(tnum_sca[bin_mask], scada.ws.values[bin_mask], statistic="std", bins=time_bins)[0]

#mean yaw
cos = binned_statistic(tnum_sca[bin_mask], np.cos(np.radians(scada.yaw.values[bin_mask])), statistic="mean", bins=time_bins)[0]
sin = binned_statistic(tnum_sca[bin_mask], np.sin(np.radians(scada.yaw.values[bin_mask])), statistic="mean", bins=time_bins)[0]
yaw_sca_avg=np.degrees(np.arctan2(sin,cos))%360

#%% Output
stats=xr.Dataset()

stats['data_avail_lid']=xr.DataArray(data_avail_lid,coords={'time':time_avg})
stats['ws_lid_avg']=xr.DataArray(ws_lid_avg,coords={'time':time_avg})
stats['ws_lid_std']=xr.DataArray(ws_lid_std,coords={'time':time_avg})
stats['wd_lid_avg']=xr.DataArray(wd_lid_avg,coords={'time':time_avg})

stats['data_avail_sca']=xr.DataArray(data_avail_sca,coords={'time':time_avg})
stats['power_sca_avg']=xr.DataArray(power_sca_avg,coords={'time':time_avg})
stats['power_sca_std']=xr.DataArray(power_sca_std,coords={'time':time_avg})
stats['ws_sca_avg']=xr.DataArray(ws_sca_avg,coords={'time':time_avg})
stats['ws_sca_std']=xr.DataArray(ws_sca_std,coords={'time':time_avg})
stats['yaw_sca_avg']=xr.DataArray(yaw_sca_avg,coords={'time':time_avg})

stats.to_netcdf(os.path.join(config['path_data'],f'{sdate}.{edate}.{turbine}.lidar.stats.nc'))