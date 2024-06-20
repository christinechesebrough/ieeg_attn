#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:26:51 2023

@author: max
"""

#%% To-Do
# Need to label spikes and other artifacts in data

import os, sys, re
import numpy as np
import mne
from itertools import compress
from pynwb import NWBHDF5IO
from helpers import interp_bad_samples, combine_left_right, prep_gaze, detect_saccades_remodnav
from tqdm import tqdm
from scipy import ndimage
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#import cv2
import scipy.interpolate as interp
import scipy.stats as stats
import scipy.signal as signal
#import temporal_response_function as trf
import fnmatch


sys.path.append('/Users/christinechesebrough/Documents/EPIPE-master/Python')
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar, filter_hfa_continuous

# After plotting raw channels and picking new ones to reject the kernel freezes
# One solution is disabling "Active support" for Matplotlib 
# (under tools -> preferences -> IPython console -> Graphics)
# And then manually setting the matplotlib backend, as described here:
# https://github.com/mne-tools/mne-python/issues/6528#issuecomment-892066104
matplotlib.use('Qt5Agg')



#%% Parameters
full_task_name = 'movies'
pipeline_name = 'preprocess_movies'
pipeline_version = 'v.0.0.0'

resample_fs = 600
notch_freqs = (60, 120, 180)

# Types of references to use in analyses
# Must be a list containing at least one of the options: "avg", "bip"
ref_types = ['avg', 'bip']

n_jobs = 16

convert_db = True

# Frequency range
freq_range = [70, 170]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100

#%% Define directoris
data_dir = '/Volumes/Expansion/Movie_data/movies_nwb_standard'
fs_dir = '/Volumes/Documents/Movie_data/anatomy'
eloc_dir = '/Volumes/Expansion/Movie_data/data/electrode_localization'
#prep_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'
prep_dir = '/Volumes/Expansion/Movie_data/movies_rest_standard'
#frame_dir = '/Users/christinechesebrough/Documents/HBML_all/data/video_frames'
#lum_dir = 'data/luminance'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'

patients = os.listdir(data_dir)
patients.sort()


et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'


#%%   
broken_nwb = [] 

patients = [patient for patient in os.listdir(data_dir) if not patient.startswith('.DS_Store')]
patients.sort()

pat = 'sub-NS135'

for pat in patients:
    pat = 'sub-NS135'

    directory_path = os.path.join(data_dir, pat)
    all_files = os.listdir(directory_path)
    
    implants = [filename for filename in all_files if os.path.isdir(os.path.join(directory_path, filename)) and not filename == '.DS_Store']
    
    for imp in implants:
        
        num_imp = int(re.findall(r'\d+', imp)[0])
        
        if num_imp == 1:
            pat_fs = pat.replace('sub-', '')
        elif num_imp >= 2:
            pat_fs = f'{pat.replace("sub-", "")}_{num_imp:02d}'

        sub_fs_dir = os.path.join(fs_dir, pat_fs)
        
        movies = os.listdir('{:s}/{:s}/{:s}'.format(data_dir, pat, imp))

        # List all movies/sessions in this directory
        #movies = [mov for mov in os.listdir(movies) if not mov.startswith('.DS_Store')]
        
 
        for mov in movies:
            # Construct the file path in a safe manner
            nwb_fname = os.path.join(data_dir, pat, imp, mov)
            
            nwb_fname = '/Users/christinechesebrough/Documents/Movies_data/RS_data_converted/NS135/B22/NS135_B22_rest.nwb'
            
            # NWB read
            try:
                io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
                nwb = io.read()
            except:
                broken_nwb.append(nwb_fname)
                continue
            
            # Get info on data in NWB file
            nwbInfo = inspectNwb(nwb)
            tsInfo = nwbInfo['timeseries']
            elecTable = nwbInfo['elecs']
            
            # Get ieeg data
            try: 
                
              # if 'ieeg' in tsInfo['name'].to_list():
                    ecogContainer = nwb.acquisition.get('ieeg')
                    fs = ecogContainer.rate
                    ecog = nwb2mne(ecogContainer,preload=False)
                
                    # Get coordinates of each electrode that has
                    try:
                        ielvis_df = read_ielvis(sub_fs_dir)
                        ch_coords = {}
                        nan_array = np.empty((3,)) * np.nan
                        for thisChn in ecog.ch_names:
                            idx = np.where(ielvis_df['label'] == thisChn)[0]
                            if len(idx) == 1:
                                xyz = np.array(ielvis_df.iloc[idx[0]]['LEPTO'])
                                ch_coords[thisChn] = xyz/1000
                            elif len(idx) == 0:
                                ch_coords[thisChn] = nan_array
                            else:
                                raise ValueError('More than 1 found!')
                    except:
                        ch_coords = {}
                        for thisChn in ecog.ch_names:
                            ch_coords[thisChn] = np.empty((3,)) * np.nan
                
                    # Create `montage` data structure as required by MNE
                    montage = mne.channels.make_dig_montage(ch_pos=ch_coords, coord_frame='mri')
                    montage.add_estimated_fiducials(pat_fs, fs_dir)
                    ecog.set_montage(montage)
                    
                    # Load audio
                    audioContainer = nwb.acquisition.get('audio')
                    fs_audio = audioContainer.rate
                    audio = audioContainer.data[:]
                    t_audio = np.arange(0, audio.shape[0]) / fs_audio
                    
            except:
                
                broken_nwb.append(nwb_fname)
                continue
            
            # Get the current sampling rate. Important for later
            orig_fs = ecog.info['sfreq']
            
            # Get the TTL pulses. Specify the name of the container with the TTL pulses
            ttl_container_name = 'TTL'
            try:
                ttls = nwb.get_acquisition(ttl_container_name).timestamps[()]
            except:
                # An analog TTL channel, convert to discrete timestamps
                ana_ttls = nwb.get_acquisition(ttl_container_name).data[()].flatten()
                ttl_rate = nwb.get_acquisition(ttl_container_name).rate
                from epipe import ana2dig
                _, ttls = ana2dig(ana_ttls, fs=ttl_rate, min_diff=0.4, return_time=True)
            
            ttl_id = nwb.acquisition['TTL'].data[:]
           
            #Preprocessing
            sub_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, neural_prep_dir)
            
            #sub_prep_dir = '/Users/christinechesebrough/Documents/HBML_all/data/movies_nwb/sub-NS135/neural_prep'
            
            if not os.path.exists(sub_prep_dir):
                os.makedirs(sub_prep_dir)
                
            #region Notch filter, down sample, add/remove bad channels by inspecting raw trace, save
            preproc_filename = '{:s}/{:s}'.format(sub_prep_dir, mov.replace('.nwb', '_prep.fif'))

           # preproc_filename = 'NS135_prep.fif'
            
            if not os.path.exists(preproc_filename):
                
                print('--->Applying notch filters and downsampling to %2.fHz' % resample_fs)
                
                # Copy the `ecog` variable and then resample and apply notch filter
                ecogPreproc = ecog.resample(resample_fs).notch_filter(notch_freqs, notch_widths=2)
                
                # High pass to remove drift would be helpfull here
                
                # Display the raw traces and mark bad channels
                nbadOrig = ecogPreproc.info['bads']
                fig = ecogPreproc.plot(show=True, block=True, remove_dc=True, duration=30.0, n_channels=32)
                
                # Save the current state of the data in the MNE format                                
                ecogPreproc.save(preproc_filename, 
                                 fmt='single', overwrite=True)
            
            else:
                ecogPreproc = mne.io.read_raw(preproc_filename, preload=True)
            
            # This loop runs all other steps on all types of references specified to use
            for ref in ref_types:
            
                print('#' * 50)
                print('Beginning processing for data using %s reference' % ref)
                print('#' * 50)
            
                # What the preprocessed filename for this reference type should be
                preprocRerefFname = '{:s}/{:s}'.format(sub_prep_dir, 
                                                       mov.replace('.nwb', '_prep_ref_{:s}.fif'.format(ref)))
            
                # Check if the preprocessed file already exists so you don't have to redo rereferncing functions
                if os.path.isfile(preprocRerefFname):
                    ecogReref = mne.io.read_raw_fif(preprocRerefFname, preload=True)
                    if 'ecogPreproc' in locals():
                        del ecogPreproc
                else:
                    if 'ecogPreproc' not in locals():
                        ecogPreproc = mne.io.read_raw_fif(preproc_filename, preload=True)
            
                    if ref == 'avg':
                        ecogReref = reref_avg(ecogPreproc)
            
                    elif ref == 'bip':
                        ecogReref = reref_bipolar(ecogPreproc)
                        
                    # Cut at first and last frame
                    ecogReref_cut = ecogReref.copy()
                    
                   # if 'Fix' in mov:
                       # ecogReref_cut.crop(tmin=ttls[0], tmax=ttls[-1])
                   # else:
                       # ecogReref_cut.crop(tmin=frame_time[0], tmax=frame_time[-1])
            
                    # Save the referenced data in MNE format
                    ecogReref_cut.save(preprocRerefFname,fmt='single',overwrite=True)
                    
                # Filter for HFA
                sub_hfa_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, hfa_dir)
                

                hfa_fname = '{:s}/{:s}'.format(sub_hfa_dir,
                                               mov.replace('.nwb', 
                                                           '_prep_ref_{:s}_hfa.fif'.format(ref)))
                
                if not os.path.exists(sub_hfa_dir):
                    os.makedirs(sub_hfa_dir)
                    
                if not os.path.exists(hfa_fname):
                    
                    # Compute HFA
                    hfa_mne = filter_hfa_continuous(ecogReref, hfa_fname, freq_range, freq_space, n_freq_bins,
                                                    convert_db, n_jobs, resample_bha_fs)
                
                        
                    # Cut at first and last frame
                  #  if 'Fix' in mov:
                   #     hfa_mne.crop(tmin=ttls[0], tmax=ttls[-1])
                   # else:
                    #    hfa_mne.crop(tmin=frame_time[0], tmax=frame_time[-1])
                    
                    # Save the cut HFA data
                    hfa_mne.save(hfa_fname,fmt='single',overwrite=True)
                    
            # Close NWB file
            io.close()