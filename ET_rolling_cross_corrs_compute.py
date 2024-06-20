#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 10:07:52 2024

@author: christinechesebrough
"""

from scipy.stats import pearsonr
import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


data_dir = '/media/christine/Expansion/ET_figures/precentral'
isc_dir = '/media/christine/Expansion/data/isc'
eloc_dir = '/media/christine/Expansion/electrode_localization'
fig_dir = '/media/christine/Expansion/ET_figures/precentral'


if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
    
patients = os.listdir(data_dir)
patients.sort()

    #%%

    
# Function to compute rolling correlation with a window size
def compute_rolling_correlation(series1, series2, window_size):
    return pd.Series(series1).rolling(window=window_size).corr(pd.Series(series2))

def cross_correlation(series1, series2, max_lag):
    s1 = pd.Series(series1)
    s2 = pd.Series(series2)
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            lag_corr = s1.shift(-lag).corr(s2)
        else:
            lag_corr = s1.corr(s2.shift(lag))
        correlations.append(lag_corr)
    return correlations


#%%

# Define the number of subplots and their arrangement
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

for id, pat in enumerate(patients):

    #pat = 'NS136'
    pat_dir = os.path.join(fig_dir, pat)  # Directory for saving patient-specific figures
    
    # Search for 'feature_df.csv' files in each patient directory
    et_files = [filename for filename in os.listdir(pat_dir) if 'feature_df' in filename and filename.endswith('.csv')]

    # Handle each matching file
    for et_file in et_files:
        full_et_path = os.path.join(pat_dir, et_file)
        try:
            eye_values = pd.read_csv(full_et_path)
            print("Processed eye tracking data for:", full_et_path)
        except Exception as e:
            print(f"Error reading {full_et_path}: {e}")
    
    #read in the ET values
    isc = eye_values ['isc_pat']
    sacc = eye_values ['saccade_rate_sliding']
    verg = eye_values ['vergence']
    
    # Define window size for rolling correlation
    rolling_window_size = 7  # Adjust the window size as needed
    
    rolling_corr_isc_verg = compute_rolling_correlation(isc, verg, rolling_window_size)
    rolling_corr_isc_sacc = compute_rolling_correlation(isc, sacc, rolling_window_size)
    rolling_corr_verg_sacc = compute_rolling_correlation(verg, sacc, rolling_window_size)
    
    rolling_corrs_df = pd.DataFrame({
    'ISC - vergence': rolling_corr_isc_verg,
    'ISC - saccade': rolling_corr_isc_sacc,
    'vergence - saccade': rolling_corr_verg_sacc})
    
    et_prep_filename = '{:s}/{:s}'.format(pat_dir, f'{pat}_rolling_corrs.csv')
    csv_filename = et_prep_filename
    rolling_corrs_df.to_csv(csv_filename, index=False) 
    
    # Plot the rolling correlations
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.suptitle(f'Rolling Correlations of ET measures for {pat}', fontsize=16)
    axes[0].plot(rolling_corr_isc_verg, color="red", label='Rolling Correlation ISC-Vergence')
    axes[1].plot(rolling_corr_isc_sacc, color="orange", label='Rolling Correlation ISC-Saccade Rate')
    axes[2].plot(rolling_corr_verg_sacc, color="green", label='Rolling Correlation Vergence-Saccade Rate')
    axes[0].set_xlabel('Time Index')
    axes[0].set_ylabel('Correlation')
    axes[0].legend()
    axes[1].legend()
    axes[2].legend()
    plot_path = os.path.join(pat_dir, f'{pat}_rollingCorrs.png')
    plt.savefig(plot_path)
    plt.show()

    # Compute cross correlations
    cross_corr_isc_verg = cross_correlation(isc, verg, 10)
    cross_corr_isc_sacc = cross_correlation(isc, sacc, 10)
    cross_corr_verg_sacc = cross_correlation(verg, sacc, 10)
    
    cross_corrs_df = pd.DataFrame({
    'ISC - vergence': cross_corr_isc_verg,
    'ISC - saccade': cross_corr_isc_sacc,
    'vergence - saccade': cross_corr_verg_sacc})
    
    et_prep_filename = '{:s}/{:s}'.format(pat_dir, f'{pat}_cross_corrs.csv')
    csv_filename = et_prep_filename
    cross_corrs_df.to_csv(csv_filename, index=False) 
    
     # Plot each correlation on the respective subplot
    lags = range(-10, 11)  # Adjust based on your actual max_lag
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(lags, cross_corr_isc_verg, label=f'Patient {pat}', alpha=0.5)
    axs[1].plot(lags, cross_corr_isc_sacc, label=f'Patient {pat}', alpha=0.5)
    axs[2].plot(lags, cross_corr_verg_sacc, label=f'Patient {pat}', alpha=0.5)
    
    titles = ['Cross Correlation ISC-Vergence', 'Cross Correlation ISC-Saccade Rate', 'Cross Correlation Vergence-Saccade Rate']
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Correlation Coefficient')
        ax.legend(loc='best')
    
    # Finalize plots and save figure
    plt.tight_layout()
    plot_path = os.path.join(pat_dir, f'{pat}_crossCorrs.png')
    plt.savefig(plot_path)
    plt.show()
 
#%%

# Define lists to store cross-correlation values for all patients
all_cross_corr_isc_verg = []
all_cross_corr_isc_sacc = []
all_cross_corr_verg_sacc = []

# Loop through each patient's data
for id, pat in enumerate(patients):
    pat_dir = os.path.join(fig_dir, pat)  # Directory for saving patient-specific figures
    
    # Search for '_rolling_corrs.csv' files in each patient directory
    corr_files = [filename for filename in os.listdir(pat_dir) if 'cross_corrs' in filename and filename.endswith('.csv')]

    # Handle each matching file
    for corr_file in corr_files:
        full_corr_path = os.path.join(pat_dir, corr_file)
        try:
            corr_dat = pd.read_csv(full_corr_path)
            cross_isc_verg = corr_dat['ISC - vergence'].values #ISC-Vergence Rolling Correlation
            cross_isc_sacc = corr_dat['ISC - saccade'].values #ISC - saccade rate rolling correlation
            cross_verg_sacc = corr_dat['vergence - saccade'].values # vergence-saccade rolling correlation
            print("Processed correlation data for:", full_corr_path)
        except Exception as e:
            print(f"Error reading {full_corr_path}: {e}")

    # Append cross-correlation values to the list
    all_cross_corr_isc_verg.append(cross_isc_verg)
    all_cross_corr_isc_sacc.append(cross_isc_sacc)
    all_cross_corr_verg_sacc.append(cross_verg_sacc)

# Convert lists to DataFrame
df_isc_verg = pd.DataFrame(all_cross_corr_isc_verg)
df_isc_sacc = pd.DataFrame(all_cross_corr_isc_sacc)
df_verg_sacc = pd.DataFrame(all_cross_corr_verg_sacc)

# Plotting Summary Figures
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Plotting ISC-Vergence Summary
axs[0].plot(df_isc_verg.T, alpha=0.5)
axs[0].set_title('Cross Correlation ISC-Vergence Summary')
axs[0].set_xlabel('Lag')
axs[0].set_ylabel('Correlation Coefficient')
axs[0].legend(patients, loc='best')

# Plotting ISC-Saccade Rate Summary
axs[1].plot(df_isc_sacc.T, alpha=0.5)
axs[1].set_title('Cross Correlation ISC-Saccade Rate Summary')
axs[1].set_xlabel('Lag')
axs[1].set_ylabel('Correlation Coefficient')
axs[1].legend(patients, loc='best')

# Plotting Vergence-Saccade Rate Summary
axs[2].plot(df_verg_sacc.T, alpha=0.5)
axs[2].set_title('Cross Correlation Vergence-Saccade Rate Summary')
axs[2].set_xlabel('Lag')
axs[2].set_ylabel('Correlation Coefficient')
axs[2].legend(patients, loc='best')

# Save Summary Figures
summary_plot_path_isc_verg = os.path.join(fig_dir, 'cross_corr_summary.png')

plt.tight_layout()
plt.savefig(summary_plot_path_isc_verg)
plt.show()