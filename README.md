# awaken_dual_doppler
Codes to process dual-Doppler stare scans at AWAKEN. 

To bulk process files:
- Run download.py with desired date range (either from editor or cmd)
- Run process_lidars.py with desired date range (either from editor or cmd)

The download.py downloads data from Wind Data Hub using doe_dap_dl from https://github.com/a2edap/dap-py.

The main code is process_lidar.py and requires lidargo from https://github.com/StefanoWind/FIEXTA/tree/angels.

To add those dependencies in conda, clone the source code on the local machine. Then open Anaconda command prompt and navigate where the setup.py lives, then:

`pip install .`

The code uses config files which can be found at configs[https://drive.google.com/drive/folders/1KRegx4QaQyNuewvYMmzH0FC8Rt4Z6pB9]
