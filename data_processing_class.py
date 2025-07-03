import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
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
        l2_datafile = xr.open_dataset(lidar2)
        l2_new = l2_datafile.assign_coords(scanID=l2_datafile.time)
        self.l2 = l2_new.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        
        # Defines uniform time distribution
        t1 = self.l1.time.min().values
        t2 = self.l1.time.max().values
        self.time = np.arange(t1, t2, np.timedelta64(1, "s"))
    
    # Find wind velocities
    # Returns 2D vectors
    def wind_velocity(self):
        # Defines transformation matrix
        row1 = [math.cos(self.el1) * math.sin(self.az1), math.cos(self.el1) * math.cos(self.az1)]
        row2 = [math.cos(self.el2) * math.sin(self.az2), math.cos(self.el2) * math.cos(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Interpolates data over time distribution
        rad_vel1 = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).sel(range=self.rg1)
        rad_vel1_int = rad_vel1.interp(time=self.time)
        rad_vel2 = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).sel(range=self.rg2)
        rad_vel2_int = rad_vel2.interp(time=self.time)
        
        # Defines radial velocity vector
        rv = np.array([rad_vel1_int, rad_vel2_int])
    
        # Solves for wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel
    
    # Find average wind velocities
    # Returns 2D vector
    def av_wind_velocity(self):      
        # Defines transformation matrix
        row1 = [math.cos(self.el1) * math.sin(self.az1), math.cos(self.el1) * math.cos(self.az1)]
        row2 = [math.cos(self.el2) * math.sin(self.az2), math.cos(self.el2) * math.cos(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines average radial velocity vector
        rad_vel1 = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).sel(range=self.rg1)
        rad_vel1_int = rad_vel1.interp(time=self.time).mean()
        rad_vel2 = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).sel(range=self.rg2)
        rad_vel2_int = rad_vel2.interp(time=self.time).mean()
        
        # Defines radial velocity vector
        rv = np.array([rad_vel1_int, rad_vel2_int])
        
        # Solves for average wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel_av = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel_av
    
    # Find wind speeds
    # Returns 1D array
    def wind_speed(self):
        wind_comp = self.wind_velocity()
        ws = np.hypot(wind_comp[[0], :], wind_comp[[1], :])
        return ws
    
    # Plot radial wind speeds across a series of ranges over time
    # Range limited to first 100 range values
    def plot_radial_speed(self):
        # Interpolates data over time distribution
        rad_vel1 = self.l1.wind_speed.where(self.l1.qc_wind_speed==0)
        rad_vel1_int = rad_vel1.interp(time=self.time)
        rad_vel2 = self.l2.wind_speed.where(self.l2.qc_wind_speed==0)
        rad_vel2_int = rad_vel2.interp(time=self.time)
        
        # Defines range limit
        lim = (0, 100)
        range_limit1 = rad_vel1_int.range[lim[0]:lim[1]]
        radial_speeds1 = rad_vel1_int.values[lim[0]:lim[1]]
        range_limit2 = rad_vel2_int.range[lim[0]:lim[1]]
        radial_speeds2 = rad_vel2_int.values[lim[0]:lim[1]]
                
        # Create heatmap of time and range against radial speed for lidar1
        fig1 = plt.figure(figsize=(15, 5))
        ax1 = fig1.subplots()
        image1 = ax1.pcolormesh(self.time, range_limit1, radial_speeds1)
        ax1.set_title("Lidar 1 Radial Wind Speed")
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylabel("Range (m)")
        fig1.colorbar(image1, label="Radial Wind Speed (m/s)")
        plt.show()
        
        # Create heatmap of time and range against radial speed for lidar2
        fig2 = plt.figure(figsize=(15, 5))
        ax2 = fig2.subplots()
        image2 = ax2.pcolormesh(self.time, range_limit2, radial_speeds2)
        ax2.set_title("Lidar 2 Radial Wind Speed")
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel("Range (m)")
        fig2.colorbar(image2, label="Radial Wind Speed (m/s)")
        plt.show()
    
    # Plot radial wind speeds at intersection point over time
    def plot_radial_intersect(self):
        # Interpolates data over time distribution
        rad_vel1 = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).sel(range=self.rg1)
        rad_vel1_int = rad_vel1.interp(time=self.time)
        rad_vel2 = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).sel(range=self.rg2)
        rad_vel2_int = rad_vel2.interp(time=self.time)
        
        # Create plot of time against lidar1 and lidar2 radial speeds
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, rad_vel1_int, ".", label="Lidar 1")
        ax.plot(self.time, rad_vel2_int, ".", label="Lidar 2")
        ax.grid(visible=True)
        ax.legend()
        ax.set_title("Radial Wind Speed at Intersection")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Radial Wind Speed (m/s)")
        plt.show()
    
    # Plot wind velocity components at intersection point over time
    def plot_components(self):  
        wind_comp = self.wind_velocity()
        
        # Create plot of time against u and v component velocities
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, wind_comp[[0],:].transpose(), ".", label="u")
        ax.plot(self.time, wind_comp[[1],:].transpose(), ".", label="v")
        ax.grid(visible=True)
        ax.legend()
        ax.set_title("Wind Components")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Wind Speed (m/s)")
        plt.show()
    
    # Plot wind speed at intersection point over time
    def plot_speed(self):
        speeds = self.wind_speed().transpose()
        
        # Create plot of time against wind speed
        fig = plt.figure(figsize=(15, 5))
        ax = fig.subplots()
        ax.plot(self.time, speeds, ".")
        ax.grid(visible=True)
        ax.set_title("Wind Speed")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Wind Speed (m/s)")
        plt.show()
    
    # Plot wind direction at intersection point over time
    # Direction measured in degrees
    def plot_direction(self):        
        wind_comp = self.wind_velocity()
        
        # Calculate wind direction angles from component data
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
    
    # Plot correlation coefficient over a series of ranges
    # Range limited to first 100 range values
    def plot_correlation(self):
        # Interpolates data over time distribution
        rad_vel1 = self.l1.wind_speed.where(self.l1.qc_wind_speed==0)
        rad_vel1_int = rad_vel1.interp(time=self.time)
        rad_vel2 = self.l2.wind_speed.where(self.l2.qc_wind_speed==0)
        rad_vel2_int = rad_vel2.interp(time=self.time)
        
        # Defines range limit
        lim = (0, 120)
        range_limit1 = rad_vel1_int.range[lim[0]:lim[1]]
        range_limit2 = rad_vel2_int.range[lim[0]:lim[1]]
        
        coef = np.zeros((lim[1], lim[1]))
        for r1 in range(lim[0], lim[1]):
            for r2 in range(lim[0], lim[1]):
                rad1 = rad_vel1_int.sel(range=rad_vel1_int.range[r1])
                rad2 = rad_vel2_int.sel(range=rad_vel2_int.range[r2])
                real = ~np.isnan(rad1+rad2)
                if np.sum(real)>30:
                    corr_array = np.array([rad1[real],rad2[real]])
                    corr_coef = np.corrcoef(corr_array)
                    coef[r1, r2] = corr_coef[0, 1]
        
        # Create heatmap of ranges against correlation coefficient
        fig = plt.figure()
        ax = fig.subplots()
        image = ax.pcolormesh(range_limit1, range_limit2, coef.T)
        ax.set_title("Correlation Coefficient")
        ax.set_xlabel("Lidar 1 Range (m)")
        ax.set_ylabel("Lidar 2 Range (m)")
        fig.colorbar(image, label="Radial Wind Speed (m/s)")
        plt.show()
        
        return coef, range_limit1, range_limit2
#%%
lidar1 = "sa5.lidar.z03.b0.20230726.002006.user5.vt.nc"
lidar2 = "sgpdlrhi2S4.b2.20230726.002007.vt.nc"
elevation1 = 1.55 * (math.pi/180)
elevation2 = 2.33 * (math.pi/180)
azimuth1 = 309.98 * (math.pi/180)
azimuth2 = 356.19 * (math.pi/180)
range1 = 2925
range2 = 1875

Wind_Data = Lidar_Dataset(lidar1, lidar2, elevation1, elevation2, azimuth1, azimuth2, range1, range2)