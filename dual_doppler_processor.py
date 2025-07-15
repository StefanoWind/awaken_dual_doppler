import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#%% Defines class for a pair of xarray lidar datasets
# Assumes datasets have equal number of ranges
class Dual_Doppler_Processing:
    def __init__(self, lidar1, lidar2, range1, range2,time_step=1,ang_tol=0.25,max_time_diff=2):
        
        #read an check angles
        if lidar1.elevation.std()<ang_tol:
            self.el1 = np.float64(lidar1.elevation.mean())
        else:
            raise ValueError("The elevation of lidar 1 is not constant")
            
        if lidar2.elevation.std()<ang_tol:
            self.el2 = np.float64(lidar2.elevation.mean())
        else:
            raise ValueError("The elevation of lidar 2 is not constant")
            
        if lidar1.azimuth.std()<ang_tol:
            self.az1 = np.float64(lidar1.azimuth.mean())
        else:
            raise ValueError("The azimuth of lidar 1 is not constant")
            
        if lidar2.azimuth.std()<ang_tol:
            self.az2 = np.float64(lidar2.azimuth.mean())
        else:
            raise ValueError("The azimuth of lidar 2 is not constant")
             
        self.rg1 = range1
        self.rg2 = range2
        
        # Reorganizes lidar1 dataset so that scanID is replaced with time
        self.l1 = lidar1
        self.l2 = lidar2
        
        # Defines uniform time distribution
        t1 = self.l1.time.min().values
        t2 = self.l1.time.max().values
        self.time = np.arange(t1, t2, np.timedelta64(time_step, "s"))
        tnum=(self.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        
        #calculate the difference between the interpolation time and the nearest available time
        tnum1=(self.l1.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        tnum1=tnum1.expand_dims({"range":self.l1.range})
        tnum1=tnum1.where(self.l1.qc_wind_speed==0)
        time_diff1=tnum1.interp(time=self.time,method="nearest")-tnum
        
        tnum2=(self.l2.time-np.datetime64('1970-01-01T00:00:00'))/np.timedelta64(1,'s')
        tnum2=tnum2.expand_dims({"range":self.l2.range})
        tnum2=tnum2.where(self.l2.qc_wind_speed==0)
        time_diff2=tnum2.interp(time=self.time,method="nearest")-tnum
        
        #interpolate radial wind speed over uniform time distribution
        self.ws1_int = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).interp(time=self.time).where(np.abs(time_diff1)<max_time_diff)
        self.ws2_int = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).interp(time=self.time).where(np.abs(time_diff2)<max_time_diff)
        
    # Finds wind velocities
    # Returns 2D vectors
    def wind_velocity(self):
        # Defines transformation matrix
        row1 = [self.cosd(self.el1) * self.sind(self.az1), self.cosd(self.el1) * self.cosd(self.az1)]
        row2 = [self.cosd(self.el2) * self.sind(self.az2), self.cosd(self.el2) * self.cosd(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines radial velocity vector
        rv1 = self.ws1_int.interp(range=self.rg1)
        rv2 = self.ws2_int.interp(range=self.rg2)
        rv = np.array([rv1, rv2])
    
        # Solves for wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel[0,:],wind_vel[1,:]
    
    # Finds average wind velocities
    # Returns 2D vector
    def av_wind_velocity(self):      
        # Defines transformation matrix
        row1 = [self.cosd(self.el1) * self.sind(self.az1), self.cosd(self.el1) * self.cosd(self.az1)]
        row2 = [self.cosd(self.el2) * self.sind(self.az2), self.cosd(self.el2) * self.cosd(self.az2)]
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
    def wind_speed_direction(self,u=None,v=None):
        if u is None and v is None:
            u,v = self.wind_velocity()
            
        ws = np.hypot(u,v)
        wd = (270-np.degrees(np.arctan2(v,u)))%360
        
        return ws, wd
    
    # Plots radial wind speeds across a series of ranges over time
    # # Ranges limited to specified ranges
    # def plot_radial_speed(self):
    #     # Defines range limit
    #     date=str(self.time[0])[:10]
    #     # Creates heatmap of time and range against radial speed for lidar1
    #     fig1 = plt.figure(figsize=(15, 8))
    #     ax1 = fig1.add_subplot(2,1,1)
    #     plt.pcolor(self.time, self.ws1_int.range,self.ws1_int,cmap="plasma")
    #     ax1.axhline(self.rg1,"--k")
    #     ax1.set_title("Lidar 1 Radial Wind Speed on {date}")
    #     ax1.set_xlabel("Time (UTC)")
    #     ax1.set_ylabel("Range [m]")
        
    #     plt.colorbar(label=f"Radial Wind Speed [m s$^{-1}$]")
    #     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        
        # # Creates heatmap of time and range against radial speed for lidar2
        # fig2 = plt.figure(figsize=(15, 5))
        # ax2 = fig2.subplots()
        # #image2 = ax2.pcolormesh(self.time, range_limit2, radial_speeds2)
        # range_T2 = np.array(list(zip(range_limit2.values)))
        # image2 = ax2.pcolormesh(self.time, range_limit2, int_f2((range_T2, self.time)))
        # ax2.set_title("Lidar 2 Radial Wind Speed")
        # ax2.set_xlabel("Time (UTC)")
        # ax2.set_ylabel("Range (m)")
        # fig2.colorbar(image2, label="Radial Wind Speed (m/s)")
        # plt.show()
    
    # # Plots radial wind speeds at intersection point over time
    # def plot_radial_intersect(self):
    #     # Interpolates data over time distribution
    #     rv1 = self.ws1_int.sel(range=self.rg1)
    #     rv2 = self.ws2_int.sel(range=self.rg2)
        
    #     # Creates plot of time against lidar1 and lidar2 radial speeds
    #     fig = plt.figure(figsize=(15, 5))
    #     ax = fig.subplots()
    #     ax.plot(self.time, rv1, ".", label="Lidar 1")
    #     ax.plot(self.time, rv2, ".", label="Lidar 2")
    #     ax.grid(visible=True)
    #     ax.legend()
    #     ax.set_title("Radial Wind Speed at Intersection")
    #     ax.set_xlabel("Time (UTC)")
    #     ax.set_ylabel("Radial Wind Speed (m/s)")
    #     plt.show()
    
    # Plots wind velocity components at intersection point over time
    def plot_velocities(self,u=None,v=None):  
        if u is None or v is None:
            u,v = self.wind_velocity()
        
        ws,wd=self.wind_speed_direction(u,v)
            
        date=str(self.time[0])[:10]
        
        # Creates plot of time against u and v component velocities
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(2,1,1)
        ax.plot(self.time, u, ".b", label="U")
        ax.plot(self.time, v, ".r", label="V")
        ax.plot(self.time, ws, ".k", label="Wind speed")
        ax.grid(visible=True)
        ax.legend()
        ax.set_title(f"Wind velocity on {date}")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(r"Wind velocity [m s$^{-1}$]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        
        ax = fig.add_subplot(2,1,2)
        ax.plot(self.time, wd, ".k")
        ax.grid(visible=True)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(r"Wind direction [$^\circ$C]")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        plt.tight_layout()
        
        return fig
    
    # Plots wind speed at intersection point over time
    # def plot_speed(self):
    #     speeds = self.wind_speed().transpose()
        
    #     # Creates plot of time against wind speed
    #     fig = plt.figure(figsize=(15, 5))
    #     ax = fig.subplots()
    #     ax.plot(self.time, speeds, ".")
    #     ax.grid(visible=True)
    #     ax.set_title("Wind Speed")
    #     ax.set_xlabel("Time (UTC)")
    #     ax.set_ylabel("Wind Speed (m/s)")
    #     plt.show()
    
    # # Plots wind direction at intersection point over time
    # # Directions measured in degrees
    # def plot_direction(self):        
    #     wind_comp = self.wind_velocity()
        
    #     # Calculates wind direction angles from component data
    #     angles = np.arctan2(wind_comp[[1],:], wind_comp[[0],:])
    #     angles = (270 - angles) % 360
        
    #     # Plots time against wind direction
    #     fig = plt.figure(figsize=(15, 5))
    #     ax = fig.subplots()
    #     ax.plot(self.time, angles.transpose(), ".")
    #     ax.set_xlabel("Time(UTC)")
    #     ax.set_ylabel("Wind Direction (degrees)")
    #     ax.set_title("Wind Direction")
    #     ax.grid(visible=True)
    #     plt.show()
    
    # Plots correlation coefficient over a series of ranges
    # Ranges limited to specified ranges
    def plot_correlation(self):
        
        #detrended RWS
        ws1_avg=self.ws1_int.resample(time='10min').mean().interp(time=self.time,method='nearest').ffill(dim='time')
        ws2_avg=self.ws2_int.resample(time='10min').mean().interp(time=self.time,method='nearest').ffill(dim='time')
        ws1_det=self.ws1_int-ws1_avg
        ws2_det=self.ws2_int-ws2_avg
        range1_sel=np.where((~np.isnan(ws1_det)).sum(dim="time").values>0)[0]
        range2_sel=np.where((~np.isnan(ws2_det)).sum(dim="time").values>0)[0]
        corr=np.zeros((len(ws1_det.range),len(ws2_det.range)))
        for i_r1 in range1_sel:
            for i_r2 in range2_sel:
                ws1=ws1_det.isel(range=i_r1).values
                ws2=ws2_det.isel(range=i_r2).values
                reals=~np.isnan(ws1+ws2)
                if np.sum(reals)>0:
                    corr[i_r1,i_r2]=np.corrcoef(ws1[reals],ws2[reals])[0,1]
        
        # Creates heatmap of ranges against correlation coefficient
        date=str(self.time[0])[:10]
        fig = plt.figure()
        ax = fig.subplots()
        plt.pcolor(self.ws1_int.range,self.ws2_int.range, corr.T,cmap='seismic',vmin=-1,vmax=1)
        ax.set_title(f"Correlation Coefficient on {date}")
        ax.set_xlabel("Lidar 1 Range (m)")
        ax.set_ylabel("Lidar 2 Range (m)")

        ax.axvline(self.rg1, color="k")
        ax.axhline(self.rg2, color="k")
        plt.colorbar(label='Correlation coefficient')
        plt.xlim([0,self.rg1+300])
        plt.ylim([0,self.rg2+300])
        plt.grid()
        
        return fig
    
    def cosd(self,angle):
        return math.cos(math.radians(angle))
     
    def sind(self,angle):
        return math.sin(math.radians(angle))
    
# #%%
# lidar1 = "C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/awaken_lidar_processing/data/awaken/sa5.lidar.z03.b0/sa5.lidar.z03.b0.20230726.002006.user5.vt.nc"
# lidar2 = "C:/Users/sletizia/OneDrive - NREL/Desktop/Main/ENDURA/awaken_lidar_processing/data/awaken/arm.lidar.sgp_s4.rhi2.b2/sgpdlrhi2S4.b2.20230726.002007.vt.nc"
# elevation1 = 1.55 * (math.pi/180)
# elevation2 = 2.33 * (math.pi/180)
# azimuth1 = 309.98 * (math.pi/180)
# azimuth2 = 356.19 * (math.pi/180)
# range1 = 2925 #actual range from A5: 2929
# range2 = 1875 #actual range from A1: 1861

# Wind_Data = Lidar_Dataset(lidar1, lidar2, elevation1, elevation2, azimuth1, azimuth2, range1, range2)
# wind_vel=Wind_Data.wind_velocity()

# fig_corr=Wind_Data.plot_correlation()