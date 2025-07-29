import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
#%% Open lidar data set and normalize lidar wind speed
lidar = xr.open_dataset("sa5.sa1.lidar.vt.c0.20230803.000000.nc")
lidar_ws_norm = lidar.ws.values / np.nanmean(lidar.ws.values)
#%% Reads csv, finds g02 info
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
#%% interpolate SCADA power and SCADA wind speed, normalize power and wind speed, calculate corrcoefs
power_interp = np.interp(lidar.time.values.astype("float64"), scada_time.astype("float64"), scada_power)
power_norm = power_interp / np.mean(power_interp)

ws_interp = np.interp(lidar.time.values.astype("float64"), scada_time.astype("float64"), scada_ws)
ws_interp[np.isnan(lidar.ws.values)] = np.nan
ws_norm = ws_interp / np.nanmean(ws_interp)

mask = ~np.isnan(lidar.ws.values)

coef1 = np.corrcoef(ws_norm[mask], power_norm[mask])
print(coef1)
coef2 = np.corrcoef(lidar_ws_norm[mask], power_norm[mask])
print(coef2)
#%% Experimenting with breaking lidar wind speed into components parallel and perpendicular to nacelle
#This will break if the maximum differnce in angles is greater than 90 degrees
angle_interp = np.interp(lidar.time.values.astype("float64"), scada_time.astype("float64"), scada_angle)
diff_angle = np.abs(angle_interp - lidar.wd.values)
parallel = lidar.ws.values * np.cos(np.deg2rad(diff_angle))
parallel_norm = parallel / np.nanmean(parallel)

coef3 = np.corrcoef(parallel_norm[mask], power_norm[mask])
print(coef3)
#%% Plots power vs wind speed
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7))

ax1.plot(lidar.time.values, power_norm, color="red", label="Normalized SCADA Active Power")
ax1.plot(lidar.time.values, ws_norm, color="blue", label="Normalized SCADA Wind Speed")
ax1.legend()
ax1.grid(visible=True)
ax1.set_title("SCADA wind speed against G02 active power on 2023-08-03")
ax1.set_xlabel("Time (UTC)")
ax1.set_ylabel("Normalized values")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

ax2.plot(lidar.time.values, power_norm, color="red", label="Normalized SCADA Active Power")
ax2.plot(lidar.time.values, lidar_ws_norm, color="blue", label="Normalized Lidar Wind Speed")
ax2.legend()
ax2.grid(visible=True)
ax2.set_title("Lidar wind speed against G02 active power on 2023-08-03")
ax2.set_xlabel("Time (UTC)")
ax2.set_ylabel("Normalized values")
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 

fig1.subplots_adjust(hspace=0.4)
plt.show()
#%% Plots parallel lidar wind speed component vs power
fig2, ax3 = plt.subplots(figsize=(18, 4))

ax3.plot(lidar.time.values, power_norm, color="red", label="Normalized SCADA Active Power")
ax3.plot(lidar.time.values, parallel_norm, color="blue", label="Normalized Lidar Wind Speed Parallel to Nacelle")
ax3.legend()
ax3.grid(visible=True)
ax3.set_title("Lidar wind speed parallel to nacelle against G02 active power on 2023-08-03")
ax3.set_xlabel("Time (UTC)")
ax3.set_ylabel("Normalized values")
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')) 

plt.show()