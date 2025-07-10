import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy as sp
#%% Defines class for a pair of xarray lidar datasets
# Assumes datasets have equal number of ranges
class Lidar_Dataset:
    def __init__(self, lidar1, lidar2, elevation1, elevation2, azimuth1, azimuth2, range1, range2):
        self.el1 = elevation1
        self.el2 = elevation2
        self.az1 = azimuth1
        self.az2 = azimuth2
        self.rg1 = range1
        self.rg2 = range2
        
        # Reorganizes lidar1 dataset so that scanID is replaced with time
        self.l1 = xr.open_dataset(lidar1)
        self.l1 = self.l1.assign_coords(scanID=self.l1.time)
        self.l1 = self.l1.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        # Reorganizes lidar2 dataset so that scanID is replaced with time
        self.l2 = xr.open_dataset(lidar2)
        self.l2 = self.l2.assign_coords(scanID=self.l2.time)
        self.l2 = self.l2.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        
        # Defines uniform time distribution
        t1 = self.l1.time.min().values
        t2 = self.l1.time.max().values
        self.time = np.arange(t1, t2, np.timedelta64(1, "s"))
        # Interpolates wind speed over uniform time distribution
        self.ws1_int = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).interp(time=self.time)
        self.ws2_int = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).interp(time=self.time)
        
    # Finds wind velocities
    # Returns 2D vectors
    def wind_velocity(self):
        # Defines transformation matrix
        row1 = [math.cos(self.el1) * math.sin(self.az1), math.cos(self.el1) * math.cos(self.az1)]
        row2 = [math.cos(self.el2) * math.sin(self.az2), math.cos(self.el2) * math.cos(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines radial velocity vector
        rv1 = self.ws1_int.sel(range=self.rg1)
        rv2 = self.ws2_int.sel(range=self.rg2)
        rv = np.array([rv1, rv2])
    
        # Solves for wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel
    
    # Finds average wind velocities
    # Returns 2D vector
    def av_wind_velocity(self):      
        # Defines transformation matrix
        row1 = [math.cos(self.el1) * math.sin(self.az1), math.cos(self.el1) * math.cos(self.az1)]
        row2 = [math.cos(self.el2) * math.sin(self.az2), math.cos(self.el2) * math.cos(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines radial velocity vector
        rv1 = self.ws1_int.sel(range=self.rg1).mean()
        rv2 = self.ws2_int.sel(range=self.rg2).mean()
        rv = np.array([rv1, rv2])
        
        # Solves for average wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel_av = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel_av
    
    # Finds wind speeds
    # Returns 1D array
    def wind_speed(self):
        wind_comp = self.wind_velocity()
        ws = np.hypot(wind_comp[[0], :], wind_comp[[1], :])
        return ws
    
    # Plots radial wind speeds across a series of ranges over time
    # Ranges limited to specified ranges
    def plot_radial_speed(self):
        # Defines range limit
        lim = (0, 100)
        range_limit1 = self.ws1_int.range[lim[0]:lim[1]]
        radial_speeds1 = self.ws1_int.values[lim[0]:lim[1]]
        range_limit2 = self.ws2_int.range[lim[0]:lim[1]]
        radial_speeds2 = self.ws2_int.values[lim[0]:lim[1]]
        
        # Defines interpolation function
        coordinates1 = (range_limit1.values, self.time)
        int_f1 = sp.interpolate.RegularGridInterpolator(coordinates1, radial_speeds1)
        coordinates2 = (range_limit2.values, self.time)
        int_f2 = sp.interpolate.RegularGridInterpolator(coordinates2, radial_speeds2)
                
        # Creates heatmap of time and range against radial speed for lidar1
        fig1 = plt.figure(figsize=(15, 5))
        ax1 = fig1.subplots()
        #image1 = ax1.pcolormesh(self.time, range_limit1, radial_speeds1)
        range_T1 = np.array(list(zip(range_limit1.values)))
        image1 = ax1.pcolormesh(self.time, range_limit1, int_f1((range_T1, self.time)))
        ax1.set_title("Lidar 1 Radial Wind Speed")
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylabel("Range (m)")
        fig1.colorbar(image1, label="Radial Wind Speed (m/s)")
        plt.show()
        
        # Creates heatmap of time and range against radial speed for lidar2
        fig2 = plt.figure(figsize=(15, 5))
        ax2 = fig2.subplots()
        #image2 = ax2.pcolormesh(self.time, range_limit2, radial_speeds2)
        range_T2 = np.array(list(zip(range_limit2.values)))
        image2 = ax2.pcolormesh(self.time, range_limit2, int_f2((range_T2, self.time)))
        ax2.set_title("Lidar 2 Radial Wind Speed")
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel("Range (m)")
        fig2.colorbar(image2, label="Radial Wind Speed (m/s)")
        plt.show()
    
    # Plots radial wind speeds at intersection point over time
    def plot_radial_intersect(self):
        # Interpolates data over time distribution
        rv1 = self.ws1_int.sel(range=self.rg1)
        rv2 = self.ws2_int.sel(range=self.rg2)
        
        # Creates plot of time against lidar1 and lidar2 radial speeds
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, rv1, ".", label="Lidar 1")
        ax.plot(self.time, rv2, ".", label="Lidar 2")
        ax.grid(visible=True)
        ax.legend()
        ax.set_title("Radial Wind Speed at Intersection")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Radial Wind Speed (m/s)")
        plt.show()
    
    # Plots wind velocity components at intersection point over time
    def plot_components(self):  
        wind_comp = self.wind_velocity()
        
        # Creates plot of time against u and v component velocities
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, wind_comp[[0],:].transpose(), ".", label="U")
        ax.plot(self.time, wind_comp[[1],:].transpose(), ".", label="V")
        ax.grid(visible=True)
        ax.legend()
        ax.set_title("Wind Components")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Wind Speed (m/s)")
        plt.show()
    
    # Plots wind speed at intersection point over time
    def plot_speed(self):
        speeds = self.wind_speed().transpose()
        
        # Creates plot of time against wind speed
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, speeds, ".")
        ax.grid(visible=True)
        ax.set_title("Wind Speed")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Wind Speed (m/s)")
        plt.show()
    
    # Plots wind direction at intersection point over time
    # Directions measured in degrees
    def plot_direction(self):        
        wind_comp = self.wind_velocity()
        
        # Calculates wind direction angles from component data
        angles = np.arctan2(wind_comp[[1],:], wind_comp[[0],:])
        angles = (270 - angles) % 360
        
        # Plots time against wind direction
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, angles.transpose(), ".")
        ax.set_xlabel("Time(UTC)")
        ax.set_ylabel("Wind Direction (degrees)")
        ax.set_title("Wind Direction")
        ax.grid(visible=True)
        plt.show()
    
    # Plots correlation coefficient over a series of ranges
    # Ranges limited to specified ranges
    def plot_correlation(self):
        
        #detrended RWS
        ws1_det = (self.ws1_int - self.ws1_int.mean(dim='time')).dropna(dim='range', how='all')
        ws2_det = (self.ws2_int - self.ws2_int.mean(dim='time')).dropna(dim='range', how='all')
        
        #expand over other range
        ws1_exp = ws1_det.rename({'range':'range1'}).expand_dims({'range': ws2_det.range}).rename({'range':'range2'}).transpose('range1','range2','time')  # (time, x, y)
        ws2_exp = ws2_det.rename({'range':'range2'}).expand_dims({'range': ws1_det.range}).rename({'range':'range1'}) # (time, x, y)
        
        # Compute correlation matrix: corr(x, y)
        numerator = (ws1_exp*ws2_exp).mean(dim='time')
        denominator = ws1_exp.std(dim='time') *  ws2_exp.std(dim='time') 
        
        corr = numerator / denominator
        
        # Creates heatmap of ranges against correlation coefficient
        fig = plt.figure()
        ax = fig.subplots()
        plt.pcolor(corr.range1,corr.range2, corr.T,cmap='seismic',vmin=-1,vmax=1)
        ax.set_title("Correlation Coefficient")
        ax.set_xlabel("Lidar 1 Range (m)")
        ax.set_ylabel("Lidar 2 Range (m)")

        ax.axvline(self.rg1, color="k")
        ax.axhline(self.rg2, color="k")
        plt.colorbar(label='Correlation coefficient')
        plt.xlim([0,self.rg1+300])
        plt.ylim([0,self.rg2+300])
        plt.grid()
        
        return fig
        
#%%
lidar1 = "C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/awaken_lidar_processing/data/awaken/sa5.lidar.z03.b0/sa5.lidar.z03.b0.20230726.002006.user5.vt.nc"
lidar2 = "C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/awaken_lidar_processing/data/awaken/arm.lidar.sgp_s4.rhi2.b2/sgpdlrhi2S4.b2.20230726.002007.vt.nc"
elevation1 = 1.55 * (math.pi/180)
elevation2 = 2.33 * (math.pi/180)
azimuth1 = 309.98 * (math.pi/180)
azimuth2 = 356.19 * (math.pi/180)
range1 = 2925 #actual range from A5: 2929
range2 = 1875 #actual range from A1: 1861

Wind_Data = Lidar_Dataset(lidar1, lidar2, elevation1, elevation2, azimuth1, azimuth2, range1, range2)
wind_vel=Wind_Data.wind_velocity()

fig_corr=Wind_Data.plot_correlation()