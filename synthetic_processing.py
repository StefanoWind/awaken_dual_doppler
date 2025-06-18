import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
#%% Choose data sets
lidar1 = xr.open_dataset("lidar_data1.nc")
lidar2 = xr.open_dataset("new_lidar.nc")
#%% Set variables
elevation1 = 5 * (math.pi/180)
elevation2 = 3 * (math.pi/180)
azimuth1 = 280 * (math.pi/180)
azimuth2 = 10 * (math.pi/180)
range1 = 1005
range2 = 495
scan = 0
#%% Define wind velocity and wind speed functions
def wind_velocity(el1=elevation1, el2=elevation2, az1=azimuth1, az2=azimuth2, rg1=range1, rg2=range2, sc=scan):
    # Defines transformation matrix
    row1 = [math.cos(el1) * math.sin(az1), math.cos(el1) * math.cos(az1), math.sin(el1)]
    row2 = [math.cos(el2) * math.sin(az2), math.cos(el2) * math.cos(az2), math.sin(el2)]
    matrix_a = np.array([row1, row2])
    
    # Defines radial velocity vector
    rad_vel1 = float(lidar1.wind_speed.sel(range=rg1, scanID=sc))
    rad_vel2 = float(lidar2.wind_speed.sel(range=rg2, scanID=sc))
    rad_vel_av1 = lidar1.wind_speed.sel(range=rg1).mean()
    rad_vel_av2 = lidar2.wind_speed.sel(range=rg2).mean()
    rv = np.array([[rad_vel1], [rad_vel2]])
    rv_av = np.array([[rad_vel_av1], [rad_vel_av2]])
    
    # Defines system of equations
    equ_matrix = np.array([[matrix_a[0, 0], matrix_a[0, 1]], [matrix_a[1, 0], matrix_a[1, 1]]])
    
    # Solves for wind velocity
    equ_matrix_inv = np.linalg.inv(equ_matrix)
    wind_vel = np.matmul(equ_matrix_inv, rv)
    wind_vel_av = np.matmul(equ_matrix_inv, rv_av)
    
    return wind_vel, wind_vel_av

def wind_speed(wv=wind_velocity()):
    ws = math.sqrt((wv[0][0, 0]**2) + (wv[0][1, 0]**2))
    ws_av = math.sqrt((wv[1][0, 0]**2) + (wv[1][1, 0]**2))
    return ws, ws_av
#%%
print(wind_speed())