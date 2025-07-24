import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy as sp
import xarray as xr

#%% Defines class for a pair of xarray lidar datasets
class Dual_Doppler_Processing:
    def __init__(self, lidar1, lidar2, range1,range2, config,logger):

        self.config = config
        self.logger=logger
        
        #read an check angles
        if lidar1.elevation.std()<config['ang_tol']:
            self.el1 = np.float64(lidar1.elevation.mean())
        else:
            raise ValueError("The elevation of lidar 1 is not constant")
            
        if lidar2.elevation.std()<config['ang_tol']:
            self.el2 = np.float64(lidar2.elevation.mean())
        else:
            raise ValueError("The elevation of lidar 2 is not constant")
            
        if lidar1.azimuth.std()<config['ang_tol']:
            self.az1 = np.float64(lidar1.azimuth.mean())
        else:
            raise ValueError("The azimuth of lidar 1 is not constant")
            
        if lidar2.azimuth.std()<config['ang_tol']:
            self.az2 = np.float64(lidar2.azimuth.mean())
        else:
            raise ValueError("The azimuth of lidar 2 is not constant")
            
        #crop data
        lidar1=lidar1.where((lidar1.range>=range1-config['extend_range'])*(lidar1.range<=range1+config['extend_range']),drop=True)
        lidar2=lidar2.where((lidar2.range>=range2-config['extend_range'])*(lidar2.range<=range2+config['extend_range']),drop=True)
        
        self.rg1 = range1
        self.rg2 = range2
        
        self.l1 = lidar1
        self.l2 = lidar2
        
        # Defines uniform time distribution
        logger.info(f"Interpolating in time with a timestep of {config['time_step']} s")
        t1 = self.l1.time.min().values
        t2 = self.l1.time.max().values
        self.time = np.arange(t1, t2, np.timedelta64(config['time_step'], "s"))
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
        self.ws1_int = self.l1.wind_speed.where(self.l1.qc_wind_speed==0).interp(time=self.time).where(np.abs(time_diff1)<config['max_time_diff'])
        self.ws2_int = self.l2.wind_speed.where(self.l2.qc_wind_speed==0).interp(time=self.time).where(np.abs(time_diff2)<config['max_time_diff'])
        
        # Interpolates missing data values for lidar1
        interp_limit = config["interp_limit"] # pixels
        logger.info("Filling missing values for lidar 1")
        valid_mask1 = ~np.isnan(self.ws1_int)
        distance1 = sp.ndimage.distance_transform_edt(~valid_mask1)
        interp_mask1 = (np.isnan(self.ws1_int)) & (distance1 <= interp_limit)
        yy1, xx1 = np.indices(self.ws1_int.shape)
        points1 = np.column_stack((yy1[valid_mask1], xx1[valid_mask1]))
        values1 = self.ws1_int.values[valid_mask1]
        interp_points1 = np.column_stack((yy1[interp_mask1], xx1[interp_mask1]))
        interpolated_values1 = sp.interpolate.griddata(points1, values1, interp_points1, method='linear')
        ws1_inpaint = self.ws1_int.values.copy()
        ws1_inpaint[interp_mask1] = interpolated_values1
        self.ws1_inpaint=xr.DataArray(ws1_inpaint,coords=self.ws1_int.coords)
        
        # Interpolates missing data values for lidar2
        logger.info("Filling missing values for lidar 2")
        valid_mask2 = ~np.isnan(self.ws2_int)
        distance2 = sp.ndimage.distance_transform_edt(~valid_mask2)
        interp_mask2 = (np.isnan(self.ws2_int)) & (distance2 <= interp_limit)
        yy2, xx2 = np.indices(self.ws2_int.shape)
        points2 = np.column_stack((yy2[valid_mask2], xx2[valid_mask2]))
        values2 = self.ws2_int.values[valid_mask2]
        interp_points2 = np.column_stack((yy2[interp_mask2], xx2[interp_mask2]))
        interpolated_values2 = sp.interpolate.griddata(points2, values2, interp_points2, method='linear')
        ws2_inpaint = self.ws2_int.values.copy()
        ws2_inpaint[interp_mask2] = interpolated_values2
        self.ws2_inpaint=xr.DataArray(ws2_inpaint,coords=self.ws2_int.coords)
        
    # Finds wind velocities
    # Returns 2D vectors
    def wind_velocity(self):
        
        # Defines transformation matrix
        self.logger.info("Dual-Doppler reconstruction")
        row1 = [self.cosd(self.el1) * self.sind(self.az1), self.cosd(self.el1) * self.cosd(self.az1)]
        row2 = [self.cosd(self.el2) * self.sind(self.az2), self.cosd(self.el2) * self.cosd(self.az2)]
        equ_matrix = np.array([row1, row2])
        
        # Defines radial velocity vector
        rv1 = self.ws1_inpaint.interp(range=self.rg1)
        rv2 = self.ws2_inpaint.interp(range=self.rg2)
        self.rv1=rv1
        self.rv2=rv2
        
        rv = np.array([rv1, rv2])
    
        # Solves for wind velocity
        equ_matrix_inv = np.linalg.inv(equ_matrix)
        wind_vel = np.matmul(equ_matrix_inv, rv)
        
        return wind_vel[0,:],wind_vel[1,:]
    
    # Finds wind speeds
    # Returns 1D array
    def wind_speed_direction(self,u=None,v=None):
        if u is None and v is None:
            u,v = self.wind_velocity()
            
        ws = np.hypot(u,v)
        wd = (270-np.degrees(np.arctan2(v,u)))%360
        
        return ws, wd
   
    # Plots wind velocity components at intersection point over time
    def plot_radial_velocities(self):  
        
        date=str(self.time[0])[:10]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(18, 10))
        
        # Creates radial speed heatmaps
        rad1 = ax1.pcolormesh(self.time, self.ws1_int.range, self.ws1_int, cmap="plasma")
        fig.colorbar(rad1, label="Radial \n "+r"Wind Speed [m s$^{-1}$]")
        ax1.set_title(f"Lidar 1 Radial Wind Speed on {date}")
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylabel("Range (m)")
        rad2 = ax2.pcolormesh(self.time, self.ws2_int.range, self.ws2_int, cmap="plasma")
        fig.colorbar(rad2, label="Radial \n "+r"Wind Speed [m s$^{-1}$]")
        ax2.set_title(f"Lidar 2 Radial Wind Speed on {date}")
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel("Range (m)")
        
        # Creates inpainted radial speed heatmaps
        inpaint1 = ax3.pcolormesh(self.time, self.ws1_inpaint.range, self.ws1_inpaint, cmap="plasma")
        fig.colorbar(inpaint1, label="Radial \n "+r"Wind Speed [m s$^{-1}$]")
        ax3.set_title(f"Inpainted Lidar 1 Radial Wind Speed on {date}")
        ax3.set_xlabel("Time (UTC)")
        ax3.set_ylabel("Range (m)")
        inpaint2 = ax4.pcolormesh(self.time, self.ws2_inpaint.range, self.ws2_inpaint, cmap="plasma")
        fig.colorbar(inpaint2, label="Radial \n "+r"Wind Speed [m s$^{-1}$]")
        ax4.set_title(f"Inpainted Lidar 2 Radial Wind Speed on {date}")
        ax4.set_xlabel("Time (UTC)")
        ax4.set_ylabel("Range (m)")
        plt.tight_layout()
        
        return fig
    
    def plot_velocities(self,u=None,v=None):  
        date=str(self.time[0])[:10]
        if u is None or v is None:
            u,v = self.wind_velocity()
        
        ws,wd=self.wind_speed_direction(u,v)
        
        # Creates plot of time against u and v component velocities
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10))
        ax1.plot(self.time, self.rv1, ".b", label="Lidar1",markersize=2)
        ax1.plot(self.time, self.rv2, ".r", label="Lidar2",markersize=2)
        ax1.grid(visible=True)
        ax1.legend()
        ax1.set_title(f"Radial wind velocity on {date}")
        ax1.set_xlabel("Time (UTC)")
        ax1.set_ylabel(r"Range [m]")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        
        ax2.plot(self.time, u, ".b", label="U",markersize=2)
        ax2.plot(self.time, v, ".r", label="V",markersize=2)
        ax2.plot(self.time, ws, ".k", label="Wind speed",markersize=2)
        ax2.grid(visible=True)
        ax2.legend()
        ax2.set_title(f"Wind velocity on {date}")
        ax2.set_xlabel("Time (UTC)")
        ax2.set_ylabel(r"Wind velocity [m s$^{-1}$]")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 

        ax3.plot(self.time, wd, ".k",markersize=2)
        ax3.grid(visible=True)
        ax3.set_xlabel("Time (UTC)")
        ax3.set_ylabel(r"Wind direction [$^\circ$]")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 
        plt.tight_layout()
        
        return fig
    
    # Plots correlation coefficient over a series of ranges
    def plot_correlation(self):
        #detrended RWS
        ws1_avg=self.ws1_inpaint.resample(time='10min').mean().interp(time=self.time,method='nearest').ffill(dim='time')
        ws2_avg=self.ws2_inpaint.resample(time='10min').mean().interp(time=self.time,method='nearest').ffill(dim='time')
        ws1_det=self.ws1_inpaint-ws1_avg
        ws2_det=self.ws2_inpaint-ws2_avg
        range1_sel=np.where((~np.isnan(ws1_det)).sum(dim="time").values>0)[0]
        range2_sel=np.where((~np.isnan(ws2_det)).sum(dim="time").values>0)[0]
        
        self.logger.info("Calculating correlation between beams")
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
        plt.pcolormesh(self.ws1_inpaint.range,self.ws2_inpaint.range, corr.T,cmap='seismic',vmin=-1,vmax=1)
        ax.set_title(f"Correlation Coefficient on {date}")
        ax.set_xlabel("Lidar 1 Range (m)")
        ax.set_ylabel("Lidar 2 Range (m)")

        ax.axvline(self.rg1, color="k")
        ax.axhline(self.rg2, color="k")
        plt.colorbar(label='Correlation coefficient')
        plt.xlim([self.rg1-self.config['extend_range'],self.rg1+self.config['extend_range']])
        plt.ylim([self.rg2-self.config['extend_range'],self.rg2+self.config['extend_range']])
        plt.grid()
        
        return fig
    
    def cosd(self,angle):
        return math.cos(math.radians(angle))
     
    def sind(self,angle):
        return math.sin(math.radians(angle))
    