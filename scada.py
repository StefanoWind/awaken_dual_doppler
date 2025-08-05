import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
from scipy.stats import linregress
from scipy.stats import binned_statistic
#%% Reads data
lidar = xr.open_dataset("sa5.sa1.lidar.vt.c0.20230803.000000.nc")

file_path = "C:/Users/zuribe/Documents/GitHub/SULI_2025/test_data"
scada_time = []
scada_power = []
scada_ws = []
scada_angle = []
for file in os.listdir(file_path):
    scada = pd.read_csv("test_data/" + file)
    time = pd.to_datetime(scada.iloc[:, 0])
    scada_time.extend(list(time))
    power = np.array(scada["PKGP1HIST01.OKWF001_KP_TurbineG2.ActivePower"])
    scada_power.extend(list(power))
    ws = np.array(scada["PKGP1HIST01.OKWF001_KP_TurbineG2.WindSpeed"])
    scada_ws.extend(list(ws))
    angle = np.array(scada["PKGP1HIST01.OKWF001_KP_TurbineG2.Nacelle_Position"])
    scada_angle.extend(list(angle))
scada_time = np.array(scada_time).astype("datetime64[ns]")
scada_power = np.array(scada_power)
scada_ws = np.array(scada_ws)
scada_angle = np.array(scada_angle)
#%% Interpolates SCADA data over lidar time frame, normalizes wind speed and power time series
lidar_ws_norm = lidar.ws.values / np.nanmean(lidar.ws.values)

power_interp = np.interp(lidar.time.values.astype("float64"), scada_time.astype("float64"), scada_power)
power_norm = power_interp / np.mean(power_interp)

ws_interp = np.interp(lidar.time.values.astype("float64"), scada_time.astype("float64"), scada_ws)
ws_interp[np.isnan(lidar.ws.values)] = np.nan
ws_norm = ws_interp / np.nanmean(ws_interp)

angle_interp = np.interp(lidar.time.values.astype("float64"), scada_time.astype("float64"), scada_angle)
#%% Finds correlation coefficients
ws_condition = ~np.isnan(lidar.ws.values) & (power_norm < 10)
lidar_ws_condition = ~np.isnan(lidar.ws.values) & (lidar.ws.values < 10)
mask = ~np.isnan(lidar.ws.values)

coef1 = np.corrcoef(ws_norm[ws_condition], power_norm[ws_condition])
print("SCADA wind speed vs SCADA power: " + str(coef1[0, 1]))
coef2 = np.corrcoef(lidar_ws_norm[lidar_ws_condition], power_norm[lidar_ws_condition])
print("Lidar wind speed vs SCADA power: " + str(coef2[0, 1]))
coef3 = np.corrcoef(lidar.wd.values[mask], angle_interp[mask])
print("SCADA nacelle orientation vs lidar wind direction: " + str(coef3[0, 1]))
#%% Time series plots
# fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 10))
fig1, (ax2, ax3) = plt.subplots(2, 1, figsize=(18, 7))

# ax1.plot(lidar.time.values, power_norm, color="red", label="Normalized SCADA Active Power")
# ax1.plot(lidar.time.values, ws_norm, color="blue", label="Normalized SCADA Wind Speed")
# ax1.legend()
# ax1.grid(visible=True)
# ax1.set_title("SCADA wind speed against G02 active power on 2023-08-03")
# ax1.set_xlabel("Time (UTC)")
# ax1.set_ylabel("Normalized values")
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax2.plot(lidar.time.values, power_norm, color="red", label="Normalized SCADA Active Power")
ax2.plot(lidar.time.values, lidar_ws_norm, color="blue", label="Normalized Lidar Wind Speed")
ax2.legend()
ax2.grid(visible=True)
ax2.set_title("Lidar wind speed against SCADA active power on 2023-08-03")
ax2.set_xlabel("Time (UTC)")
ax2.set_ylabel("Normalized values")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 

ax3.plot(lidar.time.values, angle_interp, color="red", label="Nacelle Orientation")
ax3.plot(lidar.time.values, lidar.wd.values, color="blue", label="Wind Direction")
ax3.legend()
ax3.grid(visible=True)
ax3.set_title("Lidar wind direction against SCADA nacelle orientation on 2023-08-03")
ax3.set_xlabel("Time (UTC)")
ax3.set_ylabel("Degrees")
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 

fig1.subplots_adjust(hspace=0.4)
plt.show()
#%% Linear regression plots
def plot_lin_fit(x, y, bins=50, cmap='Greys',ax=None,cax=None,legend=True):

    # Remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    # Linear regression
    slope, intercept, r_value, _, _ = linregress(x, y)
    y_fit = slope * x + intercept
    rmsd = np.sqrt(np.mean((y - y_fit)**2))
    r_squared = r_value**2

    # Plot setup
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 2D histogram
    h = ax.hist2d(x, y, bins=bins, cmap=cmap)
    if cax is not None:
        plt.colorbar(h[3], ax=ax,cax=cax, label='Counts')

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot([np.min(x),np.max(x)],[np.min(x),np.max(x)],'--b',label='1:1')
    ax.plot(x_line, slope * x_line + intercept, color='red', linewidth=2, label='Linear fit')
    
    # Stats textbox
    textstr = '\n'.join((
        f'Intercept: {intercept:.2f}',
        f'Slope: {slope:.2f}',
        f'RMSD: {rmsd:.2f}',
        r'$R^2$: {:.2f}'.format(r_squared)
    ))
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_aspect("equal")
    if legend:
        plt.legend(draggable=True)
        
plot_lin_fit(ws_interp, lidar.ws.values)
plt.title("Linear regression of SCADA wind speed and lidar wind speed")
plt.xlabel("SCADA wind speed")
plt.ylabel("Lidar wind speed")
plt.show()
plot_lin_fit(angle_interp, lidar.wd.values)
plt.title("Linear regression of nacelle orientation and lidar wind direction")
plt.xlabel("SCADA nacelle orientation")
plt.ylabel("Lidar wind direction")
plt.show()
#%%
time_bins = np.arange(np.datetime64("2023-08-03T00:00:00.859995904"), np.datetime64("2023-08-04T00:00:00.859995904"), np.timedelta64(600, "s"))
#time_bins = np.arange(lidar.time.values.min(), lidar.time.values.max(), np.timedelta64(600, "s"))
bin_mask = ~np.isnan(lidar.time.values) & ~np.isnan(lidar.ws.values)

bin_mean = binned_statistic(lidar.time.values.astype("float64")[bin_mask], lidar.ws.values[bin_mask], statistic="mean", bins=time_bins)
mean_norm = bin_mean.statistic / np.nanmean(bin_mean.statistic)

scada_mean = binned_statistic(lidar.time.values.astype("float64")[bin_mask], ws_interp[bin_mask], statistic="mean", bins=time_bins)
scada_mean_norm = scada_mean.statistic / np.nanmean(scada_mean.statistic)

power_mean = binned_statistic(lidar.time.values.astype("float64")[bin_mask], power_interp[bin_mask], statistic="mean", bins=time_bins)
power_mean_norm = power_mean.statistic / np.nanmean(power_mean.statistic)

corr_mask1 = ~np.isnan(scada_mean_norm) & ~np.isnan(power_mean_norm)
corr_mask2 = ~np.isnan(mean_norm) & ~np.isnan(power_mean_norm)

coef4 = np.corrcoef(scada_mean_norm[corr_mask1], power_mean_norm[corr_mask1])
print("SCADA 10-minute mean wind speed vs SCADA power: " + str(coef4[0, 1]))
coef5 = np.corrcoef(mean_norm[corr_mask2], power_mean_norm[corr_mask2])
print("Lidar 10-minute mean wind speed vs SCADA power: " + str(coef5[0, 1]))
#%%
time_bins = np.arange(np.datetime64("2023-08-03T00:00:00.859995904"), np.datetime64("2023-08-04T00:00:00.859995904"), np.timedelta64(600, "s"))
bin_mask = ~np.isnan(lidar.time.values) & ~np.isnan(lidar.ws.values)

bin_sd = binned_statistic(lidar.time.values.astype("float64")[bin_mask], lidar.ws.values[bin_mask], statistic="std", bins=time_bins)
sd_norm = bin_sd.statistic / np.nanmean(bin_sd.statistic)

scada_sd = binned_statistic(lidar.time.values.astype("float64")[bin_mask], ws_interp[bin_mask], statistic="std", bins=time_bins)
scada_sd_norm = scada_sd.statistic / np.nanmean(scada_sd.statistic)

power_sd = binned_statistic(lidar.time.values.astype("float64")[bin_mask], power_interp[bin_mask], statistic="std", bins=time_bins)
power_sd_norm = power_sd.statistic / np.nanmean(power_sd.statistic)

corr_mask3 = ~np.isnan(scada_sd_norm) & ~np.isnan(power_sd_norm)
corr_mask4 = ~np.isnan(sd_norm) & ~np.isnan(power_sd_norm)
#%%
fig2, (ax4, ax5) = plt.subplots(2, 1, figsize=(18, 7))

ax4.plot(lidar.time.values, lidar.ws.values, color="orange", label="Lidar wind speed")
ax4.plot(time_bins[0:-1][~np.isnan(bin_mean.statistic)], bin_mean.statistic[~np.isnan(bin_mean.statistic)], color="blue", label="10-min mean wind speed")
ax4.errorbar(time_bins[0:-1], bin_mean.statistic, bin_sd.statistic, color="blue")
ax4.legend()
ax4.grid(visible=True)
ax4.set_title("10-min average lidar wind speed against lidar wind speed on 2023-08-03")
ax4.set_xlabel("Time (UTC)")
ax4.set_ylabel("Wind speed")
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax5.plot(time_bins[0:-1][~np.isnan(power_mean.statistic)], power_mean_norm[~np.isnan(power_mean.statistic)], color="red", label="Normalized SCADA active power")
ax5.plot(time_bins[0:-1][~np.isnan(bin_mean.statistic)], mean_norm[~np.isnan(bin_mean.statistic)], color="blue", label="Normalized wind speed")
#ax5.errorbar(time_bins[0:-1], mean_norm, sd_norm, color="blue")
ax5.legend()
ax5.grid(visible=True)
ax5.set_title("10-min average lidar wind speed against 10-min average SCADA active power on 2023-08-03")
ax5.set_xlabel("Time (UTC)")
ax5.set_ylabel("Normalized values")
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig2.subplots_adjust(hspace=0.4)
plt.show()
#%%
fig4, (ax10, ax11) = plt.subplots(2, 1, figsize=(18, 7))

ax10.plot(lidar.time.values, ws_interp, color="orange", label="SCADA wind speed")
ax10.plot(time_bins[0:-1][~np.isnan(scada_mean.statistic)], scada_mean.statistic[~np.isnan(bin_mean.statistic)], color="blue", label="10-min mean wind speed")
ax10.errorbar(time_bins[0:-1], scada_mean.statistic, scada_sd.statistic, color="blue")
ax10.legend()
ax10.grid(visible=True)
ax10.set_title("10-min average SCADA wind speed against SCADA wind speed on 2023-08-03")
ax10.set_xlabel("Time (UTC)")
ax10.set_ylabel("Wind speed")
ax10.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax11.plot(time_bins[0:-1][~np.isnan(power_mean.statistic)], power_mean_norm[~np.isnan(power_mean.statistic)], color="red", label="Normalized SCADA active power")
ax11.plot(time_bins[0:-1][~np.isnan(scada_mean.statistic)], scada_mean_norm[~np.isnan(scada_mean.statistic)], color="blue", label="Normalized SCADA 10-min means")
ax11.legend()
ax11.grid(visible=True)
ax11.set_title("10-min average SCADA wind speed against 10-minute average SCADA active power on 2023-08-03")
ax11.set_xlabel("Time (UTC)")
ax11.set_ylabel("Normalized values")
ax11.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

fig4.subplots_adjust(hspace=0.4)
plt.show()