# -*- coding: utf-8 -*-
'''
Processor of lidars through LIDARGO

Inputs (both hard-coded and available as command line inputs in this order):
    sdate [%Y-%m-%d]: start date in UTC
    edate [%Y-%m-%d]: end date in UTC
    delete [bool]: whether to delete raw data
    path_config: path to general config file
    mode [str]: serial or parallel
'''
import os
cd=os.path.dirname(__file__)
import sys
import traceback
import warnings
import lidargo as lg
import numpy as np
from datetime import datetime
import yaml
from multiprocessing import Pool
from matplotlib import pyplot as plt
import xarray as xr
import logging
import re
import glob
import dual_doppler_processor as ddp

warnings.filterwarnings('ignore')

#%% Inputs

#users inputs
if len(sys.argv)==1:
    sdate='2023-07-26' #start date
    edate='2023-07-27' #end date
    replace=False#replace existing files?
    delete=False #delete input files?
    path_config=os.path.join(cd,'configs/config.yaml') #config path
    mode='serial'#serial or parallel
else:
    sdate=sys.argv[1]
    edate=sys.argv[2] 
    delete=sys.argv[3]=="True"
    delete=sys.argv[4]=="True"
    path_config=sys.argv[5]
    mode=sys.argv[6]#
    
#%% Initalization

#configs
with open(path_config, 'r') as fid:
    config = yaml.safe_load(fid)

#initialize main logger
logfile_main=os.path.join(cd,'log',datetime.strftime(datetime.now(), '%Y%m%d.%H%M%S'))+'_errors.log'
os.makedirs('log',exist_ok=True)

#%% Functions
def standardize_file(file,save_path_stand,config,logfile_main,sdate,edate,replace,delete):
    date=re.search(r'\d{8}.\d{6}',file).group(0)[:8]
    if datetime.strptime(date,'%Y%m%d')>=datetime.strptime(sdate,'%Y-%m-%d') and datetime.strptime(date,'%Y%m%d')<=datetime.strptime(edate,'%Y-%m-%d'):
        try:
            logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
            lproc = lg.Standardize(file, config=config['path_config_stand'], verbose=True,logfile=logfile)
            lproc.process_scan(replace=False, save_file=True, save_path=save_path_stand)
            if delete:
                os.remove(file)
        except:
            with open(logfile_main, 'a') as lf:
                lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error standardizing file {os.path.basename(file)}: \n")
                traceback.print_exc(file=lf)
                lf.write('\n --------------------------------- \n')
                
def dual_doppler_rec(files1,files2, elevation1, elevation2, azimuth1, azimuth2, range1, range2,save_path,logfile_main,replace):
    # try:
        # logfile=os.path.join(cd,'log',os.path.basename(file).replace('nc','log'))
    
    #concatenate daily files
    lidar1=xr.Dataset()
    lidar2=xr.Dataset()
    for file1,file2 in zip(files1,files2):
        l1=xr.open_dataset(file1)
        l1=l1.assign_coords(scanID=l1.time)
        l1=l1.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        
        if 'wind_speed' in lidar1.data_vars:
            lidar1=xr.concat([lidar1,l1],dim='time')
        else:
            lidar1=l1
            
        l2=xr.open_dataset(file2)
        l2=l2.assign_coords(scanID=l2.time)
        l2=l2.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
        
        if 'wind_speed' in lidar2.data_vars:
            lidar2=xr.concat([lidar2,l2],dim='time')
        else:
            lidar2=l2

    lproc = ddp.Dual_Doppler_Processing(lidar1,lidar2, elevation1, elevation2, azimuth1, azimuth2, range1, range2)
    wind_vel=lproc.wind_velocity()
            

        # if delete:
        #     os.remove(file)
        # except:
        #     with open(logfile_main, 'a') as lf:
        #         lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error standardizing file {os.path.basename(file)}: \n")
        #         traceback.print_exc(file=lf)
        #         lf.write('\n --------------------------------- \n')

def dates_from_files(files):
    '''
    Extract data from data filenames
    '''
    dates=[]
    for f in files:
        match = re.search( r"\b\d{8}\.\d{6}\b", os.path.basename(f))
        dates.append(match.group().split('.')[0])
    
    return dates

#%% Main
# for channel in config['channels']:
        
    #standardize all files within date range
    # files=glob.glob(os.path.join(config['path_data'],channel,config['wildcard_stand'][channel]))
    # if mode=='serial':
    #     for f in files:
    #           standardize_file(f,None,config,logfile_main,sdate,edate,replace,delete)
    # elif mode=='parallel':
    #     args = [(files[i],None, config,logfile_main,sdate,edate,replace,delete) for i in range(len(files))]
    #     with Pool() as pool:
    #         pool.starmap(standardize_file, args)
    # else:
    #     raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")

files={}
dates={}
sites=[]
for c in config['chennels_ddp']:
    channel=config['chennels_ddp'][c]    
    files[c]=glob.glob(os.path.join(config['path_data'],channel,'*nc'))
    dates[c]=dates_from_files(files[c])
    sites.append(c)

channel_save=f's{sites[0].lower()}.s{sites[1].lower()}.lidar.vt.c0'
save_path=os.path.join(config['path_data'],config['chennels_ddp'][c].split('/')[0],channel_save)
dates_common=list(set(dates[sites[0]]) & set(dates[sites[1]]))    
if mode=='serial':
    for date in dates_common:
        files1=np.array(files[sites[0]])[np.array(dates[sites[0]])==date]
        files2=np.array(files[sites[1]])[np.array(dates[sites[1]])==date]
        dual_doppler_rec(files1,files2,
                         config['elevation'][sites[0]],config['elevation'][sites[1]],
                         config['azimuth'][sites[0]],config['azimuth'][sites[1]],
                         config['range'][sites[0]],config['range'][sites[1]],
                         save_path,logfile_main,replace)
          
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
        
