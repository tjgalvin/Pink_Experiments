"""
Small helper script to scrap away the FITS header information
and extract just the data array. Will place them into a numpy array
which can then be dumped to disk. A lot easier to transfer
tens of fils instead of tens of thousands

This script differs slightly from numpy_dumpy.py as I am passing
through as a argument the catalog to open. Note that it also expects
the `Images` folder to be in the working directory. I find it helpful
to make a symlink - ln -s `<path_to_images>` `Images`
"""
import os
from tqdm import tqdm
import numpy as np 
import pandas as pd 
from astropy.io import fits

import sys
if len(sys.argv) != 2:
    print(sys.argv)
    print('Expecting name of csv file to load')
    sys.exit()

df = pd.read_csv(sys.argv[1])

ds = 25000
dump_count = 0
f_dir = 'Images/first'
w_dir = 'Images/wise_reprojected'
found = 0 

first = []
wise = []

for i, r in tqdm(df.iterrows()):
    cleared = False
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
            cleared = True

    except Exception as e:
        print(e)

if not cleared:
    print('Writing remaining items...')
    first_arr = np.array(first)
    wise_arr = np.array(wise)
    print(first_arr.shape)
    np.save(f'Dumps/first_arr_{dump_count}', first_arr.astype('f'))
    np.save(f'Dumps/wise_arr_{dump_count}', wise_arr.astype('f'))
