#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:43:26 2024

Extract peak alpha frequencies in brain region during rest (computed in IAF compute) and filter movie recording
by a freq window around that peak frequency.
Reduce dims so it can be concatenated with ET features and alpha power, calculated using eye_neuralstate_corr

@author: christine
"""

from scipy.stats import pearsonr
import os
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import matplotlib.pyplot as plt

#%%

computer = "mac"

if computer == "mac":
    data_dir = '/Volumes/Expansion/Movie_data/movie_prep_good_ET_2'
    mne_data_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'
    isc_dir = '/Volumes/Expansion/Movie_data/data/isc'
    eloc_dir = '/Volumes/Expansion/Movie_data/data/electrode_localization'
    fig_dir = '/Volumes/Expansion/Movie_data/alpha_features'
    peak_dir = '/Volumes/Expansion/Movie_data/movies_rest_standard'
    et_dir = '/Volumes/Expansion/Movie_data/ET_features/A1/derived_data/'
    movie_subs_table = pd.read_excel('/Volumes/Expansion/Movie_data/data/movie_subs_table.xlsx')

    
elif computer == "PC":
    data_dir = '/media/christine/Expansion/Movie_data/movie_prep_good_ET_2'
    mne_data_dir = '/media/christine/Expansion/Movie_data/movies_prep_standard'
    isc_dir = '/media/christine/Expansion/Movie_data/data/isc'
    eloc_dir = '/media/christine/Expansion/Movie_data/data/electrode_localization'
    fig_dir = '/media/christine/Expansion/Movie_data/alpha_features'
    peak_dir = '/media/christine/Expansion/Movie_data/movies_rest_standard'
    et_dir = '/media/christine/Expansion/Movie_data/ET_features/Inf_Par/derived_data/'
    movie_subs_table = pd.read_excel('/media/christine/Expansion/Movie_data/data/movie_subs_table.xlsx')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

vid = 'DespMeEng'
vid = 'despicable_me'
ref = 'bip'
region = 'A1'

#%% Load ISC data 

data = np.load(isc_dir + '/isc_gaze_position_time.npz')   

#is time_isc time or isc...
time_isc = data['time_isc']
fs_isc = 1 / np.mean(np.diff(time_isc))   

patients_isc = data['patients']
patients = os.listdir(data_dir)
patients.sort()
#if 'NS135' in patients:
#    patients.remove('NS135')
#if 'NS153' in patients:
#    patients.remove('NS153')

#%%
for pat in patients:

    pat_dir = os.path.join(data_dir, pat)
    fig_patient_dir = os.path.join(fig_dir, pat)  # Directory for saving patient-specific figures
    if not os.path.exists(fig_patient_dir):
        os.makedirs(fig_patient_dir)

    # Electrode location
    if pat == 'NS174_02':
        elecs_subs = movie_subs_table[movie_subs_table.SubID == 'NS174']
    else: 
        elecs_subs = movie_subs_table[movie_subs_table.SubID == pat]
        
    # Load LFP and compute alpha power in inferior parietal channels
    
    lfp_pat_dir = '{:s}/{:s}/Neural_prep'.format(mne_data_dir, pat)
    
    lfp_files = os.listdir(lfp_pat_dir)
    
    lfp_files = list(compress(lfp_files, ['_ref_bip' in f for f in lfp_files]))
    #lfp_files = list(compress(lfp_files, ['Fixation' in f for f in lfp_files]))
    vid_list = list(compress(lfp_files, [vid in f for f in lfp_files]))

    vid_file = vid_list[0]
    lfp_file = vid_file
            
    mne_data = mne.io.read_raw('{:s}/{:s}'.format(lfp_pat_dir, lfp_file))
    mne_data.drop_channels(mne_data.info['bads'])
    
    labels = mne_data.ch_names
      
    ip_contacts = elecs_subs[elecs_subs.DK_Atlas == 'transversetemporal'].Contact.values
    #ip_contacts = elecs_subs[elecs_subs.DK_Atlas == 'lateraloccipital'].Contact.values
    
    if len(ip_contacts) == 0:
        pass
    else:
        # Find in bipolar channel labels
        labels_split = [l.split('-') for l in labels]
        
        #double check this is subsetting/indexing correctly
        idx_ip = np.unique(np.concatenate([np.where([np.sum([ld == ls 
                                                             for ls in lf]) 
                                                     for lf in labels_split])[0] 
                                               for ld in ip_contacts]))
        idx_ip = np.in1d(np.arange(len(labels)), idx_ip)
        labels_ip = list(compress(labels, idx_ip))       
        
        # Get data
        lfp = mne_data.get_data()
        
        fs_lfp = mne_data.info['sfreq']
        time_lfp = mne_data.times
            
        # Get LFP in inferior parietal lobe
        lfp_ip = lfp[idx_ip, :]
  
       # plt.plot(time_lfp,lfp_ip[0,:])
       # plt.plot(time_isc, isc_time_gaze.T, color='tab:Gray')
            
        #labels_ip = list(compress(labels, idx_ip))
        
        # Load peak freq data
        peak_pat_dir = '{:s}/{:s}'.format(peak_dir, pat)
        peak_files = os.listdir(peak_pat_dir)
        peak_file = [file for file in peak_files if 'peak_alpha_frequencies_fooof_by_electrodeavg' in file][0]
        peak_dat = pd.read_csv('{:s}/{:s}'.format(peak_pat_dir, peak_file))
        
        # Filter rows where DK_Atlas is "inferiorparietal"
        ip_data = peak_dat.loc[peak_dat['DK_Atlas'] == 'transversetemporal']
        
        # Calculate highest value in "Peak Alpha Frequency (Hz)"
        max_peak_alpha = ip_data['Peak Alpha Frequency (Hz)'].max()
        
        # Calculate average value in "Peak Alpha Frequency (Hz)"
        mean_peak_alpha = ip_data['Peak Alpha Frequency (Hz)'].mean()
        
        # Print the results
        print(f'Highest Peak Alpha Frequency (Hz) in {region}: {max_peak_alpha:.2f}')
        print(f'Average Peak Alpha Frequency (Hz) in {region}: {mean_peak_alpha:.2f}')
        
        if np.isnan(mean_peak_alpha):
            print(f"Skipping patient {pat} due to lack of alpha peak")
            pass
        else:
                    
            # Create a channel info structure for the inferior parietal electrodes
            info = mne.create_info(ch_names=labels_ip, sfreq=fs_lfp, ch_types='eeg')
       
            lower_bound = mean_peak_alpha - 1
            upper_bound = mean_peak_alpha + 1
            
            # Convert the raw signal into a Raw object
            raw_sig = mne.io.RawArray(lfp_ip, info)
            
            # Plot the raw signal
           # raw_sig.plot(show=True, title=f"Raw Signal - {pat}")
    
            # Bandpass and hilbert
            sos = signal.butter(5, [lower_bound,upper_bound], btype='bandpass', output='sos', fs=fs_lfp)
            #  sos = signal.butter(5, [lower_bound,upper_bound], btype='bandpass', output='sos', fs=fs_lfp)
            band_alpha = signal.sosfiltfilt(sos, lfp_ip, axis=1)
            pow_alpha = np.abs(signal.hilbert(band_alpha, axis=1))
            
            # Convert the band-pass filtered signal into a Raw object
            #band_alpha = mne.io.RawArray(band_alpha, info)
            
            # Convert the Hilbert-transformed power signal into a Raw object
            #pow_alpha = mne.io.RawArray(pow_alpha, info)
            
            # Plot the band-pass filtered signal
           # print(f"Plotting band-pass filtered signal (Alpha Band) for {pat}...")
           # band_alpha.plot(show=True, title=f"Alpha Band Filtered Signal - {pat}")
    
            # Plot the Hilbert transformed alpha power signal
          #  print(f"Plotting Alpha Power signal (Hilbert Transform) for {pat}...")
          #  pow_alpha.plot(show=True, title=f"Alpha Power (Hilbert Transform) - {pat}")
             
            # Transform the filtered signal so that it matches the length of fs_isc
            sos = signal.butter(5, 0.5*fs_isc, btype='lowpass', output='sos', fs=fs_lfp)
            pow_alpha = signal.sosfiltfilt(sos, pow_alpha, axis=1)
            f = interpolate.interp1d(time_lfp, pow_alpha)
            pow_alpha = f(time_isc)
            
            # Convert the shortened signal into a Raw object
            pow_alpha_isc = mne.io.RawArray(pow_alpha, info)
            
            # Plot the shortened signal
        #    pow_alpha_isc.plot(show=True, title=f"Shortened alpha power signal- {pat}")
        
            alpha_ip = pow_alpha
            labels = labels_ip
            alpha_ip = alpha_ip.T
            
            data_list = []  
            variable_names = []
            
            if 'alpha_ip' in locals() and 'labels_ip' in locals():
            # Check if alpha_ip is 1-dimensional
                if alpha_ip.ndim == 1:
                    data_list.append(alpha_ip)  # Directly append since it's already 1D
                    variable_names.append(f'IAP_{region}_{labels_ip[0]}')  # Assuming labels_ip has at least one label
                else:
                    # If alpha_ip is 2-dimensional, proceed as before
                    for i, label in enumerate(labels_ip):
                        data_list.append(alpha_ip[:, i])  # Add each channel as a separate column
                        variable_names.append(f'IAP_{region}_{label}')
                        
          
            data_matrix = np.column_stack(data_list)  # stack data columns side by side
                    
            # Create a DataFrame with the peak alpha values and variable names
            IAP_df = pd.DataFrame(data_matrix, columns=variable_names)
            
            pat_et_dir = os.path.join(et_dir, pat)
            
            # Search for 'feature_df.csv' files in each patient directory
            et_files = [filename for filename in os.listdir(pat_et_dir) if 'feature_df' in filename and filename.endswith('.csv')]
            
            # Handle each matching file
            for et_file in et_files:
                full_et_path = os.path.join(pat_et_dir, et_file)
                try:
                    eye_values = pd.read_csv(full_et_path)
                    print("Processed eye tracking data for:", full_et_path)
                    
                    # Concatenate IAP_df and eye_values horizontally
                    concatenated_df = pd.concat([eye_values, IAP_df], axis=1)
                    
                    # Save the concatenated DataFrame
                    concatenated_df.to_csv(os.path.join(pat_et_dir, f'{pat}_ET_features_with_IAP_in_{region}.csv'), index=False)
                except Exception as e:
                    print(f"Error reading {full_et_path}: {e}")
                    continue
            
           