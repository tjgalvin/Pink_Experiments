"""
Small helper script to scrap away the FITS header information
and extract just the data array. Will place them into a numpy array
which can then be dumped to disk. A lot easier to transfer
tens of fils instead of tens of thousands
"""
import os
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from astropy.io import fits

df = pd.read_csv('FIRST_Cata_Images.csv')

ds = 25000
dump_count = 0
f_dir = 'Images/first'
w_dir = 'Images/wise_reprojected'
found = 0 

first = []
wise = []

for i, r in tqdm(df.iterrows()):
    fn = r['filename']
    
    try:
        with fits.open(f'{f_dir}/{fn}') as hdu1:
            first_data = hdu1[0].data.copy()

        with fits.open(f'{w_dir}/{fn}') as hdu1:
            wise_data = hdu1[0].data.copy()


        first.append(first_data)
        wise.append(wise_data)

        if len(first) == ds:
            print(f'DS reached {len(first)}')
            first_arr = np.array(first)
            wise_arr = np.array(wise)
            print(first_arr.shape)
            np.save(f'Dumps/first_arr_{dump_count}', first_arr.astype('f'))
            np.save(f'Dumps/wise_arr_{dump_count}', wise_arr.astype('f'))

            dump_count += 1
            first = []
            wise = []

    except Exception as e:
        print(e)
