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
from datetime import datetime,timedelta
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
    sdate='2023-10-24' #start date
    edate='2023-10-25' #end date
    replace=False#replace existing files?
    delete=False #delete input files?
    path_config=os.path.join(cd,'configs/config.yaml') #config path
    mode='serial'#serial or parallel
else:
    sdate=sys.argv[1]
    edate=sys.argv[2] 
    replace=sys.argv[3]=="True"
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
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

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
                
def dual_doppler_rec(files,sites,date,config,save_path,logfile_main,replace):
    try:
        #compose filename
        filename=f'{os.path.basename(save_path)}.{date}.000000.nc'
        os.makedirs(save_path,exist_ok=True)
        
        #create logger
        logger = logging.getLogger("my_logger")
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(logging.DEBUG)  # Set level to DEBUG to cap
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(os.path.join(save_path,filename.replace('nc','log')))
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        #find files
        files1=sorted(np.array(files[sites[0]])[np.array(dates[sites[0]])==date])
        files2=sorted(np.array(files[sites[1]])[np.array(dates[sites[1]])==date])
        if len(files1)==0 or len(files2)==0:
            raise BaseException(f'No files on {date}')
        logger.info(f"Building {filename} from {len(files1)} files from lidar 1 and {len(files2)} files from lidar 2")
        
        #concatenate daily files
        lidar1=xr.Dataset()
        for file1 in files1:
            try:
                l1=xr.open_dataset(file1)
                if len(l1.beamID)==1:
                    l1=l1.assign_coords(scanID=l1.time)
                    l1=l1.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
                    
                    if 'wind_speed' in lidar1.data_vars:
                        lidar1=xr.concat([lidar1,l1],dim='time')
                    else:
                        lidar1=l1
            except:
                with open(logfile_main, 'a') as lf:
                    lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error reading file {file1}: \n")
                    traceback.print_exc(file=lf)
                    lf.write('\n --------------------------------- \n')
                    
        lidar2=xr.Dataset()            
        for file2 in files2:
            try:
                l2=xr.open_dataset(file2)
                if len(l2.beamID)==1:
                    l2=l2.assign_coords(scanID=l2.time)
                    l2=l2.drop_vars(["time", "beamID"]).squeeze(drop=True).rename({"scanID": "time"})
                    
                    if 'wind_speed' in lidar2.data_vars:
                        lidar2=xr.concat([lidar2,l2],dim='time')
                    else:
                        lidar2=l2
            except:
                with open(logfile_main, 'a') as lf:
                    lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error reading file {file2}: \n")
                    traceback.print_exc(file=lf)
                    lf.write('\n --------------------------------- \n')
    
        lproc = ddp.Dual_Doppler_Processing(lidar1,lidar2,config['range'][sites[0]],config['range'][sites[1]],config,logger)
        u,v=lproc.wind_velocity()
        ws,wd=lproc.wind_speed_direction(u,v)
        
        #output
        if not os.path.isfile(os.path.join(save_path,filename)) or replace==True:
            output=xr.Dataset()
            output['u']=xr.DataArray(u,coords={'time':lproc.time},attrs={'units':'m/s','long_name':'high-frequency zonal (W-E) wind velocity'})
            output['v']=xr.DataArray(v,coords={'time':lproc.time},attrs={'units':'m/s','long_name':'high-frequency meridional (S-N) wind velocity'})
            output['ws']=xr.DataArray(ws,coords={'time':lproc.time},attrs={'units':'m/s','long_name':'high-frequency horizontal wind speed'})
            output['wd']=xr.DataArray(wd,coords={'time':lproc.time},attrs={'units':'degrees','long_name':'high-frequency wind direction'}) 
            output.attrs={'comment':f'Created by Zoe Uribe and Stefano Letizia on {datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S")}',
                          'description':'Dual-Doppler velocity reconstruction from stare files. Vertical velocity is neglected.',
                          'contact':'stefano.letizia@nrel.gov'}
            
            output.to_netcdf(os.path.join(save_path,filename))
            
            #plots
            lproc.plot_radial_velocities()
            plt.savefig(os.path.join(save_path,filename).replace(".nc",".rad.vel.png"))
            
            lproc.plot_velocities(u,v)
            plt.savefig(os.path.join(save_path,filename).replace(".nc",".vel.png"))
            
            lproc.plot_correlation()
            plt.savefig(os.path.join(save_path,filename).replace(".nc",".corr.png"))

    except:
        with open(logfile_main, 'a') as lf:
            lf.write(f"{datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')} - ERROR - Error creating dual Doppler file {filename}: \n")
            traceback.print_exc(file=lf)
            lf.write('\n --------------------------------- \n')

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

#standardize all files within date range
for c in config['channels']:
    channel=config['channels'][c]
    files=glob.glob(os.path.join(config['path_data'],channel,config['wildcard_stand'][c]))
    if mode=='serial':
        for f in files:
              standardize_file(f,None,config,logfile_main,sdate,edate,replace,delete)
    elif mode=='parallel':
        args = [(files[i],None, config,logfile_main,sdate,edate,replace,delete) for i in range(len(files))]
        with Pool() as pool:
            pool.starmap(standardize_file, args)
    else:
        raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

#dual-Doppler reconstruction
files={}
dates={}
sites=[]
for c in config['chennels_ddp']:
    channel=config['chennels_ddp'][c]    
    files[c]=glob.glob(os.path.join(config['path_data'],channel,'*nc'))
    dates[c]=dates_from_files(files[c])
    sites.append(c)
    
dates_sel = [datetime.strftime((datetime.strptime(sdate,'%Y-%m-%d') + timedelta(days=i)),'%Y%m%d') for i in range((datetime.strptime(edate,'%Y-%m-%d')  - datetime.strptime(sdate,'%Y-%m-%d')).days)]

channel_save=f'{config["vt_site"]}.vtower.z01.c0'
save_path=os.path.join(config['path_data'],config['chennels_ddp'][c].split('/')[0],channel_save)
if mode=='serial':
    for date in dates_sel:
        dual_doppler_rec(files,sites,date,config,save_path,logfile_main,replace)
elif mode=='parallel':
    args = [(files,sites,dates_sel[i],config,save_path,logfile_main,replace) for i in range(len(dates_sel))]
    with Pool() as pool:
        pool.starmap(dual_doppler_rec, args)
else:
    raise BaseException(f"{mode} is not a valid processing mode (must be serial or parallel)")        
