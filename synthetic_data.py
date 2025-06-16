# -*- coding: utf-8 -*-
"""
Generate synthetic dual-Doppler data
"""

import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import warnings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

warnings.filterwarnings('ignore')
plt.close('all')

#%% Main

#scan geometry [deg]
ele1=5
ele2=3
azi1=280
azi2=10

#range [m]
r1=1000
r2=500
dr=30

#size
Nr=200
N=10000

#wind field [m/s]
U_avg=10
V_avg=5
W_avg=0
U_std=1
V_std=0.8
W_std=0.5

#%% Initialization

#build range
r=np.arange(Nr)*dr+dr/2

#generate velocity 
U=np.random.normal(U_avg,U_std,(len(r),N))
V=np.random.normal(V_avg,V_std,(len(r),N))
W=np.random.normal(W_avg,W_std,(len(r),N))

#%% Main

#virtual lidar
u_los1=np.cos(np.radians(ele1))*np.cos(np.radians(90-azi1))*U+\
       np.cos(np.radians(ele1))*np.sin(np.radians(90-azi1))*V+\
       np.sin(np.radians(ele1))*W
       
u_los2=np.cos(np.radians(ele2))*np.cos(np.radians(90-azi2))*U+\
       np.cos(np.radians(ele2))*np.sin(np.radians(90-azi2))*V+\
       np.sin(np.radians(ele2))*W

#%% Output
output1=xr.Dataset()
output1['wind_speed']=xr.DataArray(data=u_los1,coords={'range':r,'scanID':np.arange(N)})
output1.to_netcdf(os.path.join(cd,'data/sx.lidar.z01.a0.20250616.000000.nc'))

output2=xr.Dataset()
output2['wind_speed']=xr.DataArray(data=u_los2,coords={'range':r,'scanID':np.arange(N)})
output1.to_netcdf(os.path.join(cd,'data/sx.lidar.z02.a0..20250616.000000.nc'))


