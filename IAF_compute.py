#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:13:22 2024

@author: christine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 12:37:42 2024

@author: max + christine

Correlate eye movement based ISC measures with neural signals suspected to
index attentional state changes 
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
from fooof import FOOOF
from fooof.plts.spectra import plot_spectrum

#import qgrid
#import pandasgui


data_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'
#isc_dir = '/Volumes/Expansion/ISC_new'
eloc_dir = '/Volumes/Expansion/Movie_data/data/electrode_localization'
fig_dir = '/Volumes/Expansion/Movie_data/movies_rest_standard'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    
patients = os.listdir(data_dir)
patients.sort()   


#%%

data_dir = '/media/christine/Expansion/Movie_data/movies_prep_standard'
#isc_dir = '/Volumes/Expansion/ISC_new'
eloc_dir = '/media/christine/Expansion/Movie_data/data/electrode_localization'
fig_dir = '/media/christine/Expansion/Movie_data/movies_rest_standard'
#'/media/christine/Expansion/movies_results_rest'


#%% Electrode table
#movie_subs_table = pd.read_excel('/Volumes/Expansion/movie_subs_table.xlsx')
movie_subs_table = pd.read_excel('/media/christine/Expansion/Movie_data/data/movie_subs_table.xlsx')

#%%

ref_type = "avg_ref"
#ref_type = "bip_ref"

for pat in patients:
    # Set patient ID and directories
    pat_dir = os.path.join(data_dir, pat)
    fig_patient_dir = os.path.join(fig_dir, pat)  # Directory for saving patient-specific figures
    
    # Electrode location
    if pat == 'NS174_02':
        elecs_subs = movie_subs_table[movie_subs_table.SubID == 'NS174']
    else:
        elecs_subs = movie_subs_table[movie_subs_table.SubID == pat]
    
    # Load EEG data
    cx_pat_dir = '{:s}/{:s}/Neural_prep'.format(data_dir, pat)
    cx_files = os.listdir(cx_pat_dir)
    # Filter files based on the reference type
    if ref_type == 'avg_ref':
        filtered_files = list(compress(cx_files, ['resting_fixation_run-1_prep_ref_avg' in f for f in cx_files]))
    elif ref_type == 'bip_ref':
        filtered_files = list(compress(cx_files, ['resting_fixation_run-1_prep_ref_bip' in f for f in cx_files]))
    else:
        raise ValueError("Invalid reference type. Must be 'avg_ref' or 'bip_ref'")

    vid_file = cx_files[0]
    cx_file = vid_file
    mne_data = mne.io.read_raw('{:s}/{:s}'.format(cx_pat_dir, cx_file), preload=True)
    mne_data.drop_channels(mne_data.info['bads'])
    labels = mne_data.ch_names
    
    # Select contacts based on anatomical regions
    cx_contacts = elecs_subs[(elecs_subs.AparcAseg_Atlas != 'Right-Cerebral-White-Matter') &
                             (elecs_subs.AparcAseg_Atlas != 'Left-Cerebral-White-Matter')].Contact.values
    
    #select contacts only in IP    
    #cx_contacts = elecs_subs[elecs_subs.DK_Atlas == 'inferiorparietal'].Contact.values

    if len(cx_contacts) == 0:
        pass
    else:
        
        if not os.path.exists(fig_patient_dir):
            os.makedirs(fig_patient_dir)
            
        # Find channel labels
        idx_cx = np.unique([i for i, label in enumerate(labels) if any(contact in label for contact in cx_contacts)])
        labels_cx = list(compress(labels, np.in1d(np.arange(len(labels)), idx_cx)))
        
        if len(labels_cx) == 0:
            pass
        else: 
                    
            # Get all data
            eeg = mne_data.get_data()
            fs_eeg = mne_data.info['sfreq']
            time_eeg = mne_data.times
        
            # Get data across cortex
            cx_data = eeg[idx_cx, :]
        
            # Normalize
            cx_data = stats.zscore(cx_data, axis=1)
        
            # Bandpass filter
            sos = signal.butter(5, [7, 13], btype='bandpass', output='sos', fs=fs_eeg)
            band_alpha = signal.sosfiltfilt(sos, cx_data, axis=1)
        
            # Extract peak alpha frequency using Welch's method
            freqs, psd = signal.welch(band_alpha, fs_eeg, nperseg=fs_eeg*2)  # PSD using Welch's method
            alpha_band = (freqs >= 7) & (freqs <= 13)
            peak_alpha_freqs_welch = freqs[np.argmax(psd[:, alpha_band], axis=1) + np.where(alpha_band)[0][0]]
        
            # Print peak alpha frequencies using Welch's method
            for i, label in enumerate(labels_cx):
                print(f'Channel: {label}, Peak Alpha Frequency (Welch): {peak_alpha_freqs_welch[i]:.2f} Hz')
        
            # Optional: Save the peak alpha frequencies to a file
            peak_alpha_df_welch = pd.DataFrame({'Channel': labels_cx, 'Peak Alpha Frequency (Hz)': peak_alpha_freqs_welch})
            peak_alpha_df_welch.to_csv(os.path.join(fig_patient_dir, 'peak_alpha_frequencies_welch.csv'), index=False)
        
            # Plot the power spectral density of all channels within the alpha range using Welch's method
            plt.figure(figsize=(12, 8))
            for i in range(psd.shape[0]):
                plt.plot(freqs[alpha_band], psd[i, alpha_band], label=labels_cx[i])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (dB)')
            plt.title(f'Power Spectral Density (7-13 Hz) - Welch for {pat}')
            plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
            plt.tight_layout()
            plt.savefig(os.path.join(fig_patient_dir, f'{pat}_psd_alpha_band_welch.png'))
            plt.close()
        
            # Initialize lists to store FOOOF models and peak frequencies
            fooof_models = []
            peak_alpha_freqs_fooof = []
        
            # Loop over channels
            for i, label in enumerate(labels_cx):
                # Initialize and fit the FOOOF model
                fm = FOOOF(peak_width_limits=[1, 6], max_n_peaks=6, min_peak_height=0.1, aperiodic_mode='fixed')
                fm.fit(freqs[alpha_band], psd[i, alpha_band])
        
                # Get the peak frequency in the alpha band
                if fm.peak_params_.shape[0] > 0:
                    peak_alpha_freq = fm.peak_params_[0, 0]  # Get the frequency of the first peak
                else:
                    peak_alpha_freq = np.nan  # If no peaks are found, set as NaN
        
                # Append results to lists
                fooof_models.append(fm)
                peak_alpha_freqs_fooof.append(peak_alpha_freq)
                #fm.plot()
        
                # Print the FOOOF results for each channel
                print(f'Channel: {label}, Peak Alpha Frequency (FOOOF): {peak_alpha_freq:.2f} Hz')
        
            # Create DataFrame for peak alpha frequencies with DK_Atlas
            peak_alpha_df_fooof = pd.DataFrame({'Channel': labels_cx, 'Peak Alpha Frequency (Hz)': peak_alpha_freqs_fooof})
            
            # Merge with elecs_subs by matching the Channel with the index in elecs_subs
            elecs_subs = elecs_subs.set_index('Contact')
            peak_alpha_df_fooof['DK_Atlas'] = peak_alpha_df_fooof['Channel'].map(elecs_subs['DK_Atlas'])
        
            # Group by 'DK_Atlas' and compute the average peak alpha frequency
            grouped_df = peak_alpha_df_fooof.groupby('DK_Atlas')['Peak Alpha Frequency (Hz)'].mean().reset_index()
        
            # Plot the average peak alpha frequencies for each 'DK_Atlas' group
            plt.figure(figsize=(12, 6))
            bars = plt.bar(grouped_df['DK_Atlas'], grouped_df['Peak Alpha Frequency (Hz)'])
            plt.xlabel('DK Atlas Regions')
            plt.ylabel('Average Peak Alpha Frequency (Hz)')
            plt.title(f'Average Peak Alpha Frequencies by DK Atlas Regions for {pat} ({ref_type})')
            plt.xticks(rotation=90)  # Rotate x-axis labels if there are many regions
            plt.tight_layout()
            
            # Add the value of each average on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom') 
            plt.savefig(os.path.join(fig_patient_dir, f'PAF_FOOOF_byRegion_{pat}_{ref_type}'))
            plt.show()
            
            # Save the peak alpha frequencies (FOOOF) to a file
            # electrode_level
            peak_alpha_df_fooof.to_csv(os.path.join(fig_patient_dir, f'peak_alpha_frequencies_fooof_by_electrode{ref_type}.csv'), index=False)
            
            # region_level
            grouped_df.to_csv(os.path.join(fig_patient_dir,f'peak_alpha_frequencies_fooof_by_region{ref_type}.csv'),index =False)
