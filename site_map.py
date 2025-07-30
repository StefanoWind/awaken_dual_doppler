# -*- coding: utf-8 -*-
"""
Plot site map with beam schematic
"""

import os
cd=os.path.dirname(__file__)
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import utm
import xarray as xr
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['savefig.dpi'] = 500
plt.close('all')

#%% Inputs
source_layout='data/20250225_AWAKEN_layout.nc'

#site A1
lat_A1=36.362326
lon_A1=-97.405111
alt_A1=329#[m]
H_A1=4.2#[m]
dr_A1=30#[m]
azi_A1=356.19#[deg]
ele_A1=2.33#[deg]

#site A5
lat_A5=36.361693
lon_A5=-97.381415
alt_A5=324#[m]
H_A5=1#[m]
dr_A5=30#[m]
azi_A5=309.98#[deg]
ele_A5=1.55#[deg]

#target
target='G02'
alt_target=319#[m]
x_upstream=-2#[D]
ar=0.5

#graphics
rmax=5000

#%% Functions
def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))
    
def disk(x0,y0,z0,D):
    th=np.arange(0,360)
    x=cosd(th)*D/2+x0
    z=sind(th)*D/2+z0
    y=x*0+y0
    
    plt.plot(x,y,z,'.k',alpha=0.5,markersize=1)
    
def cart2spher(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    alpha = np.arctan2(y, x)%360   # azimuth
    beta = np.arccos(z / r)            # elevation
    return r, alpha, beta

#%% Initialization
Turbines=xr.open_dataset(source_layout,group='turbines')
Topography=xr.open_dataset(source_layout,group='topography')
Sites=xr.open_dataset(source_layout,group='ground_sites')

X_topo,Y_topo=np.meshgrid(Topography['x_utm'].values,Topography['y_utm'].values)
Z_topo=np.ma.masked_invalid(Topography['elevation'].values).T

#location A1
x_A1,y_A1,zone_A1_1,zone_A1_2=utm.from_latlon(lat_A1, lon_A1)
z_A1=alt_A1+H_A1

#location A5
x_A5,y_A5,zone_A5_1,zone_A5_2=utm.from_latlon(lat_A5, lon_A5)
z_A5=alt_A5+H_A5

#location G02
D=Turbines['Diameter'][Turbines.name==target].values[0]
H=Turbines['Hub height'][Turbines.name==target].values[0]
x_target=Turbines['x_utm'][Turbines.name==target].values
y_target=Turbines['y_utm'][Turbines.name==target].values+x_upstream*D
z_target=alt_target+H

#height turbines
sel=~np.isnan(Z_topo.ravel())*(np.abs(X_topo.ravel()-x_target)<rmax)*(np.abs(Y_topo.ravel()-y_target)<rmax)
sel_T=(np.abs(Turbines['x_utm'].values.ravel()-x_target)<rmax)*(np.abs(Turbines['y_utm'].values.ravel()-y_target)<rmax)
x_T=Turbines['x_utm'].values.ravel()[sel_T]
y_T=Turbines['y_utm'].values.ravel()[sel_T]
z_T = griddata((X_topo.ravel()[sel], Y_topo.ravel()[sel]), Z_topo.ravel()[sel], (x_T, y_T), method='linear')

#%% Main

#A1->target
rho_A1,alpha_A1,beta_A1=cart2spher(x_target-x_A1, y_target-y_A1, z_target-z_A1)

#A5->target
rho_A5,alpha_A5,beta_A5=cart2spher(x_target-x_A5, y_target-y_A5, z_target-z_A5)

#sampling points
r_A1=np.arange(0+dr_A1/2,rmax,dr_A1)
r_A5=np.arange(0+dr_A5/2,rmax,dr_A5)

x_beam_A1=x_A1+r_A1*cosd(90-azi_A1)*cosd(ele_A1)
y_beam_A1=y_A1+r_A1*sind(90-azi_A1)*cosd(ele_A1)
z_beam_A1=z_A1+r_A1*sind(ele_A1)

x_beam_A5=x_A5+r_A5*cosd(90-azi_A5)*cosd(ele_A5)
y_beam_A5=y_A5+r_A5*sind(90-azi_A5)*cosd(ele_A5)
z_beam_A5=z_A5+r_A5*sind(ele_A5)

#%% Plots
plt.close('all')

fig=plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
sel_x=np.abs(X_topo[0,:]-x_target)<rmax
sel_y=np.abs(Y_topo[:,0]-y_target)<rmax
surf = ax.plot_surface(X_topo[sel_y,:][:,sel_x]-x_target, Y_topo[sel_y,:][:,sel_x]-y_target, Z_topo[sel_y,:][:,sel_x]-z_target, cmap='copper',linewidth=0, antialiased=False,alpha=0.5)
plt.plot(x_A1-x_target,y_A1-y_target,z_A5-z_target,'xg',markersize=10,zorder=10)
plt.plot(x_A5-x_target,y_A5-y_target,z_A5-z_target,'xb',markersize=10,zorder=10)



for x,y,z in zip(x_T-x_target,y_T-y_target,z_T-z_target):
    plt.plot([x,x],[y,y],[z,z+H],color='k',linewidth=1, zorder=10)
    plt.plot([x,x+D/2*np.cos(np.radians(30))],[y,y],[z+H,z+H-D/2*np.sin(np.radians(30))],color='k', zorder=10)
    plt.plot([x,x-D/2*np.cos(np.radians(30))],[y,y],[z+H,z+H-D/2*np.sin(np.radians(30))],color='k', zorder=10)
    plt.plot([x,x],[y,y],[z+H,z+H+D/2],color='k', zorder=10)

plt.plot(x_beam_A1-x_target,y_beam_A1-y_target,z_beam_A1-z_target,'.g',markersize=3,
         label=r"$\rho="+str(int(rho_A1))+r"$ m, $\theta="+str(np.round(azi_A1,2))+r"^\circ$ , $\beta="+str(np.round(ele_A1,2))+r"^\circ$",zorder=10)
plt.plot(x_beam_A5-x_target,y_beam_A5-y_target,z_beam_A5-z_target,'.b',markersize=3,
         label=r"$\rho="+str(int(rho_A5))+r"$ m, $\theta="+str(np.round(azi_A5,2))+r"^\circ$ , $\beta="+str(np.round(ele_A5,2))+r"^\circ$",zorder=10)

plt.plot(0,0,0,'.r',markersize=10,label='Target',zorder=10)

plt.xlim([-rmax,rmax])
plt.ylim([-rmax,rmax])
ax.set_xlabel('W-E [m]',labelpad=20)
ax.set_ylabel('S-N [m]',labelpad=20)
ax.set_zlabel('Height [m]')
ax.view_init(90,-90)

ax.set_aspect('equal')
ax.set_zticks([-100,0,100])
plt.legend()
ax.view_init(26, -60)

#zoom
fig=plt.figure(figsize=(18,10))
ax = fig.add_subplot(111, projection='3d')
plt.plot(x_target,y_target,z_target,'.r',markersize=10)
for x,y,z in zip(x_T,y_T,z_T+H):
    if np.abs(x-x_target)<rmax/5 and np.abs(y-y_target)<rmax/5:
        disk(x,y,z,D)

plt.plot(x_beam_A1,y_beam_A1,z_beam_A1,'.g',markersize=1)
plt.plot(x_beam_A5,y_beam_A5,z_beam_A5,'.b',markersize=1)

plt.xlim([-rmax/5+x_target,rmax/5+x_target])
plt.ylim([-rmax/5+y_target,rmax/5+y_target])
ax.set_xlabel('W-E [m]')
ax.set_ylabel('S-N [m]')
ax.set_zlabel('Height ASL [m]')
ax.set_aspect("equal")

