#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:07:52 2024

@author: max + christine

Correlate eye movement based ISC measures with neural signals suspected to
index attentional state changes 
"""
from scipy.stats import pearsonr
import os
from itertools import compress
import numpy as np
import pandas as pd
from scipy import stats, signal, interpolate,correlate
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

#import qgrid
#import pandasgui

vid = 'despicable_me_english'
ref = 'bip'
# Feature of eye movement data ISC of gaze position and gaze variation ('position', 'variation')
# Eye vergence ('vergence')
#eye_feature = 'vergence'
# Region of interest Occipital-Parieta Alpha 'parietal'; posterior cingulate HFA 'pcc'
roi = 'pcc'

data_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'
isc_dir = '/Volumes/Expansion/Movie_data/data/isc'
mne_data_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'
elec_dir = '/Volumes/Expansion/Movie_data/data/electrode_localization'
fig_dir = '/Volumes/Expansion/Movie_data/new_verg_all'
movie_subs_table = pd.read_excel('/Volumes/Expansion/Movie_data/data/movie_subs_table.xlsx')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    
good_ET_pats = ['NS127_02',
 'NS135',
 'NS136',
 'NS137',
 'NS138',
 'NS140',
 'NS153',
 'NS154',
 'NS164',
 'NS166',
 'NS174_02']

n_perm = 100
fs_eye = 300


       
#%% Load ISC data 

data = np.load(isc_dir + '/isc_gaze_position_time.npz')   

#is time_isc time or isc...
time_isc = data['time_isc']
patients_isc = data['patients']
isc_time = data['isc_time_gaze']

fs_isc = 1 / np.mean(np.diff(time_isc))   

patients_isc = data['patients']
#patients = os.listdir(data_dir)
#patients.sort()

patients = good_ET_pats
patients.sort()

    #%%
import pandas as pd
import numpy as np

def calculate_pvalues(df):
    # Ensure column names are strings to avoid mixed type comparisons
    df.columns = df.columns.map(str)
    
    # Initialize DataFrame to store p-values
    pvalues = pd.DataFrame(index=df.columns, columns=df.columns)
    
    # Iterate over each pair of columns
    for r in df.columns:
        for c in df.columns:
            # Calculate p-value for the Pearson correlation coefficient
            pvalues.loc[r, c] = pearsonr(df[r], df[c])[1]  
    
    return pvalues.astype(float)

def mark_significance(pval):
    if pval < 0.001:
        return '***'
    #elif pval < 0.01:
    #    return '**'
    #elif pval < 0.05:
    #    return '*'
    else:
        return ''


#%%

region = 'all'
for pat in patients:
    pat_dir = os.path.join(data_dir, pat)
    fig_patient_dir = os.path.join(fig_dir, pat)  # Directory for saving patient-specific figures
    if not os.path.exists(fig_patient_dir):
        os.makedirs(fig_patient_dir)
        
    # identify patient array in isc matrix
    idx_pat = np.in1d(patients_isc, pat)
    
    #load isc for patient
    # select row corresponding to patient
    isc_pat = isc_time[idx_pat]

    # Load vergence data
    
    pat_dir = '{:s}/{:s}'.format(data_dir, pat)
    verg_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
    verg_file = [file for file in verg_files if 'despicable_me_english_run-1_et_prep.csv' in file][0]

    print('Loading data for patient {:s} ...'.format(pat))
    
    # read csv with vergence data
    verg_dat = pd.read_csv('{:s}/Eye_prep/{:s}'.format(pat_dir, verg_file))
    vis_fd = verg_dat['vis_fd_interp'].values #interpolated focal displacement values
    gaze_disp = verg_dat['dva_gaze_disp_x_interp'] #interpolated dva horizontal (x) disparity values
    verg_t = verg_dat['time'].values # time values for vergence data
    
    # Extract and rescale vergence data
    
    t_verg = verg_t #rename time variable from vergence data to be consistent with previous code
   # vergence = vis_fd  #assign one of the vergence values to vergence variable
    vergence = gaze_disp

    # Interpolate gaps in time axis, ensuring t_res starts and ends within t_verg range
    t_start_res, t_end_res = t_verg[0], t_verg[-1]  # Ensures that t_res is within t_verg's bounds
    t_res = np.arange(t_start_res, t_end_res, 1/fs_eye)
    
    # Interpolate vergence to match the new t_res, ensuring it's within the original data range
    f = interpolate.interp1d(t_verg, vergence, bounds_error=False, fill_value="extrapolate")
    vergence = f(t_res)
    t_verg = t_res

    # Load pupil data to align time axis of vergence to the start and end times of the pupil data...why??
    eye_pat_dir = '{:s}/{:s}/Eye_prep'.format(mne_data_dir, pat)
    eye_files = os.listdir(eye_pat_dir)
    eye_files = list(compress(eye_files, ['_et_prep' in f for f in eye_files]))
    eye_list = list(compress(eye_files, [vid in f for f in eye_files]))
    
    eye_file = eye_list[0]
    eye_data = np.load('{:s}/{:s}'.format(eye_pat_dir, eye_file))
    t_start = eye_data['t_pupil'][0]
    t_end = eye_data['t_pupil'][-1]
    print(t_start)
    print(t_end)
    
    # creating boolean associated with t_start and t_end from pupil file
    idx_vid = np.logical_and(t_verg> t_start, t_verg < t_end)
    
    #mapping this to time dim of vergence file
    t_verg = t_verg[idx_vid]
    vergence = vergence[idx_vid]
    
    #filtering t_verg (vergence) to boolean array from pupil time
    t_verg = t_verg - t_verg[0]
    
    # Saccade rate calculated with sliding window
    # Define the window length and step size (in seconds)
    window_length = 5  # 1 second window
    step_size = 2.5    # 75% overlap
    
    # Calculate the number of steps/windows
    total_time = t_verg[-1] - t_verg[0]  # Total duration of the signal
    num_steps = int(np.floor((total_time - window_length) / step_size) + 1)
    
    # Initialize an array to hold the saccade rate for each sliding window
    verg_sliding = np.zeros(num_steps)
    
    # Time vector for the center of each window
    t_verg_sliding = np.linspace(window_length / 2, total_time - window_length / 2, num=num_steps)
    
    # Slide the window across the signal and calculate the rate
    for i in range(num_steps):
        window_start = i * step_size
        window_end = window_start + window_length
        # Sum saccades within the current sliding window
        verg_sliding[i] = np.mean(vergence[(t_verg >= window_start) & (t_verg < window_end)])
    
     # Interpolate saccade_rate to match time_isc
    f_verg_sliding = interpolate.interp1d(t_verg_sliding, verg_sliding, kind='linear', fill_value="extrapolate")
    verg_interpolated_sliding = f_verg_sliding(time_isc)
         

    
    # Load and calculate saccade rate
            
    pat_dir = '{:s}/{:s}'.format(data_dir, pat)
    
    et_files = os.listdir('{:s}/Eye_prep/'.format(pat_dir))
    et_file = [file for file in et_files if 'despicable_me_english_run-1_et_prep.npz' in file][0]

    print('Loading data for patient {:s} ...'.format(pat))

    #load .npz file with saccade onset times
    et_data = np.load('{:s}/Eye_prep/{:s}'.format(pat_dir, et_file))
    
    #extract time of this ET array
    t = et_data['t_gaze']

    #boolean array corresponding to start and end times of movie (pupil) data   
    idx_vid = np.logical_and(t> t_start, t < t_end)
    t=t[idx_vid]
    t -= t[0]
    
    #extract saccade onset times
    saccade_onset_t = et_data['saccade_onset_t']
    
    #extract saccade onset times and convert to array of 1s and 0s within time 
    is_saccade_onset = np.zeros(len(t), dtype=bool) 
    
    for onset_time in saccade_onset_t:
    # Find the closest index in t for each saccade onset time, within the movie duration
        if onset_time >= t_start and onset_time <= t_end:
            idx = np.argmin(np.abs(t - (onset_time - t_start)))
            is_saccade_onset[idx] = True
    
    saccade_onset_int = is_saccade_onset.astype(int)

    
   # Saccade rate calculated with sliding window
    # Define the window length and step size (in seconds)
    window_length = 5  # 1 second window
    step_size = 2.5    # 75% overlap
    
    # Calculate the number of steps/windows
    total_time = t[-1] - t[0]  # Total duration of the signal
    num_steps = int(np.floor((total_time - window_length) / step_size) + 1)
    
    # Initialize an array to hold the saccade rate for each sliding window
    saccade_rate_sliding = np.zeros(num_steps)
    
    # Time vector for the center of each window
    t_sacc_rate_sliding = np.linspace(window_length / 2, total_time - window_length / 2, num=num_steps)
    
    # Slide the window across the signal and calculate the rate
    for i in range(num_steps):
        window_start = i * step_size
        window_end = window_start + window_length
        # Sum saccades within the current sliding window
        saccade_rate_sliding[i] = np.sum(is_saccade_onset[(t >= window_start) & (t < window_end)])
    
    # Interpolate saccade_rate to match time_isc
    f_sacc_rate_sliding = interpolate.interp1d(t_sacc_rate_sliding, saccade_rate_sliding, kind='linear', fill_value="extrapolate")
    saccade_rate_interpolated_sliding = f_sacc_rate_sliding(time_isc)
        

    isc_pat_test = isc_pat.T
    isc_pat_test = np.squeeze(isc_pat_test)
#    isc_pat_test = isc_pat

    # plot vergence and saccade rate
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time_isc,saccade_rate_interpolated_sliding,c='orange',label ='saccade rate')
    ax.plot(time_isc,verg_interpolated_sliding*10,c = 'red',label = 'sliding gaze disparity') 
    ax.plot(time_isc,isc_pat_test*50,c='blue',label ='ISC')
    plt.xlabel('Time (s)')
    plt.ylabel('Rate / Interpolated Value')
    plt.legend()
    plt.title(f'Horizontal Gaze Disparity & Saccade Rate for {pat}', fontsize=16)
    plot_path = os.path.join(fig_patient_dir, f'{pat}_saccade_vergence.png')
    plt.savefig(plot_path)
    plt.show()


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
    
   # lfp_file = vid_file.replace('aic_ref_wm_hfa', 'ref_bip'.format(ref))
        
    mne_data = mne.io.read_raw('{:s}/{:s}'.format(lfp_pat_dir, lfp_file))
    mne_data.drop_channels(mne_data.info['bads'])
    
    labels = mne_data.ch_names
    
    # Define the list of strings to exclude
    exclude_strings = ['unknown', 'bankssts']

    # Filter the contacts where DK_Atlas is not in the exclude list
    ip_contacts = elecs_subs[~elecs_subs.DK_Atlas.isin(exclude_strings)].Contact.values

   # ip_contacts = elecs_subs[elecs_subs.DK_Atlas == 'precentral'].Contact.values
   # ip_contacts = elecs_subs[(elecs_subs.DK_Atlas == 'caudalmiddlefrontal') | (elecs_subs.DK_Atlas == 'rostralmiddlefrontal')].Contact.values

    
    if len(ip_contacts) == 0:
        pass
    else:
        # Find in bipolar channel labels
        labels_split = [l.split('-') for l in labels]
        
        idx_ip = np.unique(np.concatenate([np.where([np.sum([ld == ls 
                                                             for ls in lf]) 
                                                     for lf in labels_split])[0] 
                                               for ld in ip_contacts]))
        idx_ip = np.in1d(np.arange(len(labels)), idx_ip)
        
        # Get data
        lfp = mne_data.get_data()
        fs_lfp = mne_data.info['sfreq']
        time_lfp = mne_data.times
            
        # Get LFP in inferior parietal lobe
        lfp_ip = lfp[idx_ip, :]
        
       # plt.plot(time_lfp,lfp_ip[0,:])
       # plt.plot(time_isc, isc_time_gaze.T, color='tab:Gray')

            
        # Normalize
       # lfp_ip = stats.zscore(lfp_ip, axis=1)
            
        labels_ip = list(compress(labels, idx_ip))
            
        # Bandpass and hilbert
        sos = signal.butter(5, [7,13], btype='bandpass', output='sos', fs=fs_lfp)
        band_alpha = signal.sosfiltfilt(sos, lfp_ip, axis=1)
        pow_alpha = np.abs(signal.hilbert(band_alpha, axis=1))
        
       # Create a channel info structure for the inferior parietal electrodes
       # info = mne.create_info(ch_names=labels_ip, sfreq=fs_lfp, ch_types='eeg')
            
      # Convert the band-pass filtered signal into a Raw object
       # band_alpha = mne.io.RawArray(band_alpha, info)
      
      # Convert the Hilbert-transformed power signal into a Raw object
        #pow_alpha = mne.io.RawArray(pow_alpha, info)
      
      # Plot the band-pass filtered signal
     #   print(f"Plotting band-pass filtered signal (Alpha Band) for {pat}...")
      #  band_alpha.plot(show=True, title=f"Alpha Band Filtered Signal - {pat}")
  
      # Plot the Hilbert transformed alpha power signal
       # print(f"Plotting Alpha Power signal (Hilbert Transform) for {pat}...")
       # pow_alpha.plot(show=True, title=f"Alpha Power (Hilbert Transform) - {pat}")
        
        sos = signal.butter(5, 0.5*fs_isc, btype='lowpass', output='sos', fs=fs_lfp)
        pow_alpha = signal.sosfiltfilt(sos, pow_alpha, axis=1)
        f = interpolate.interp1d(time_lfp, pow_alpha)
        
        pow_alpha = f(time_isc)
        alpha_ip = pow_alpha
        labels = labels_ip
        alpha_ip = alpha_ip.T
        
# concatenating values and correlating
    data_list = [verg_interpolated_sliding, saccade_rate_interpolated_sliding, isc_pat_test]  
    variable_names = ['vergence', 'saccade_rate_sliding', 'isc_pat']
     
        #CHANGED
    if 'alpha_ip' in locals() and 'labels_ip' in locals():
    # Check if alpha_ip is 1-dimensional
        if alpha_ip.ndim == 1:
            data_list.append(alpha_ip)  # Directly append since it's already 1D
            variable_names.append(f'alpha_{region}_{labels_ip[0]}')  # Assuming labels_ip has at least one label
        else:
            # If alpha_ip is 2-dimensional, proceed as before
            for i, label in enumerate(labels_ip):
                data_list.append(alpha_ip[:, i])  # Add each channel as a separate column
                variable_names.append(f'alpha_{region}_{label}')
    
    data_matrix = np.column_stack(data_list)  # stack data columns side by side
    
    # Create the DataFrame with the corrected variable names
    df = pd.DataFrame(data_matrix, columns=variable_names)
    df.to_csv(os.path.join(fig_patient_dir, f'{pat}_correlation_data.csv'), index=False)
    
    et_prep_filename = '{:s}/{:s}'.format(fig_patient_dir, f'{pat}_feature_df.csv')
                                
    csv_filename = et_prep_filename
    df.to_csv(csv_filename, index=False)  
            
    # Calculate the correlation matrix
    correlation_matrix = df.corr()
    
    # Display the correlation matrix
    print(correlation_matrix)
    
    # Visualization of the correlation matrix 

    calculate_pvalues(df)
#
    pvalues_matrix = calculate_pvalues(df)

    #annot = correlation_matrix
    annot = correlation_matrix.applymap('{:.2f}'.format) + pvalues_matrix.applymap(mark_significance)
    
    # Dynamic figure size
    num_variables = len(correlation_matrix.columns)
    fig_size_width = max(12, num_variables * 1.2)  # Adjust the width scaling factor as needed
    fig_size_height = max(10, num_variables * 0.8)  # Adjust the height scaling factor as needed
    
    # Visualization and saving the correlation matrix plot
    plt.figure(figsize=(fig_size_width, fig_size_height))
    ax = sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', fmt="", linewidths=.5, vmin=-1, vmax=1)
    
    plt.title(f'Correlation Matrix with Significance for Patient {pat}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=10)
    plt.tight_layout()  # Adjust the padding between and around subplots to make everything fit within the figure area
    plt.show()
    
    # Save the plot
    plot_path = os.path.join(fig_patient_dir, f'{pat}_correlation_matrix.png')
    plt.savefig(plot_path, bbox_inches='tight')  # 'bbox_inches' ensures that the whole plot is saved, including labels that might otherwise be cut off
    
    plt.close()
    print('saving data in {:s} ...'.format(fig_patient_dir))
        
    if 'alpha_ip' in locals():
        del alpha_ip



   


