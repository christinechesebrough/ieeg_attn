#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:09:29 2024

Script for calculating and plotting the differences in eye features (isc, vergence, saccade rate) based on indices 
derived from individual alpha peak frequency windows

Previous steps:
Eye features and alpha power are derived and transformed using eye_neuralstate_corr 
and then IAF values are added using filter_by_peak_alpha

@author: christine
"""

from scipy.stats import pearsonr
import os
from scipy import stats

from itertools import compress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.stats import ttest_ind


#%%

# Define a function to add mean and SD lines
def add_mean_sd_lines(data, ax):
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axhline(mean_val, color='green', linestyle='--')
    ax.axhline(mean_val + std_val, color='blue', linestyle=':')
    ax.axhline(mean_val - std_val, color='orange', linestyle=':')

#%%

computer = "mac"

if computer == "mac":
    data_dir = '/Volumes/Expansion/Movie_data/ET_features/A1/derived_data'
    fig_dir = '/Volumes/Expansion/Movie_data/ET_features/A1/IAP'
    
elif computer == "PC":
    data_dir = '/media/christine/Expansion/Movie_data/ET_features/Inf_Par/derived_data'
    fig_dir = '/media/christine/Expansion/Movie_data/ET_features/Inf_Par/IAP'

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

region = 'A1'

print(f"Data directory: {data_dir}")
print(f"Figure directory: {fig_dir}")
#%%
region = 'Lat_Occ'

# Ensure the fig_dir exists
os.makedirs(fig_dir, exist_ok=True)

print(f"Data directory: {data_dir}")
print(f"Figure directory: {fig_dir}")

#%%
sd_mult = 1

patients = [entry for entry in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, entry))]
patients.sort()
if 'NS135' in patients:
    patients.remove('NS135')
if 'NS136' in patients:
    patients.remove('NS136')
if 'NS153' in patients:
    patients.remove('NS153')

# Determine the number of rows needed for 3 columns
num_patients = len(patients)
num_cols = 3
num_rows = (num_patients + num_cols - 1) // num_cols

# Create a single summary figure with subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
fig.suptitle(f'Comparison of Int and Ext means for Peak Alpha all patients in {region} for {sd_mult}')

for idx, pat in enumerate(patients):
    row = idx // num_cols
    col = idx % num_cols

    pat_dir = os.path.join(data_dir, pat)
    pat_fig_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(pat_fig_dir):
        os.makedirs(pat_fig_dir)

    # Search for 'feature_df.csv' files in each patient directory
    et_files = [filename for filename in os.listdir(pat_dir) if 'ET_features_with_IAP' in filename and filename.endswith('.csv')]

    # Handle each matching file
    for et_file in et_files:
        full_et_path = os.path.join(pat_dir, et_file)
        try:
            eye_values = pd.read_csv(full_et_path)
            print("Processed eye tracking data for:", full_et_path)
        except Exception as e:
            print(f"Error reading {full_et_path}: {e}")
            continue

    # Read in the ET values
    isc = eye_values['isc_pat']
    sacc = eye_values['saccade_rate_sliding']
    verg = eye_values['vergence']

    eye_cols = {'isc_pat', 'saccade_rate_sliding', 'vergence'}

    if len(eye_values.columns) == 3:
        print("Skipping the code block as there are only 3 columns in eye_values.columns.")
    else:
        IAP_cols = [col for col in eye_values.columns if 'IAP' in col]
        # Initialize lists to collect aggregated values for each metric
        isc_int_values, isc_ext_values = [], []
        sacc_int_values, sacc_ext_values = [], []
        verg_int_values, verg_ext_values = [], []

        # Process each 'IAP' column
        for elec in IAP_cols:
            # Calculate the mean and standard deviation of the column
            col_data = eye_values[elec].values
            mean_value = col_data.mean()
            sd_value = col_data.std()

            # Create dynamic variable names for mean and standard deviation
            mean_var_name = f'mean_{elec}'
            sd_var_name = f'sd_{elec}'

            # Assign the mean and standard deviation values to dynamic variables in the globals() dictionary
            globals()[mean_var_name] = mean_value
            globals()[sd_var_name] = sd_value
            print(f'{mean_var_name} = {globals()[mean_var_name]}, {sd_var_name} = {globals()[sd_var_name]}')

            int_idx = np.where(col_data > mean_value + sd_mult*sd_value)[0]
            ext_idx = np.where(col_data <= mean_value - sd_mult*sd_value)[0]

            # Collect aggregated values for isc, sacc, and verg
            isc_int_values.extend(isc[int_idx])
            isc_ext_values.extend(isc[ext_idx])
            sacc_int_values.extend(sacc[int_idx])
            sacc_ext_values.extend(sacc[ext_idx])
            verg_int_values.extend(verg[int_idx])
            verg_ext_values.extend(verg[ext_idx])

        # Conduct a two-sample t-test comparing the aggregated values
        t_test_results = {
            'isc_pat': stats.ttest_ind(isc_int_values, isc_ext_values),
            'saccade_rate_sliding': stats.ttest_ind(sacc_int_values, sacc_ext_values),
            'vergence': stats.ttest_ind(verg_int_values, verg_ext_values)
        }

        # Generate a bar graph visualizing these comparisons
        labels = ['isc_pat', 'saccade_rate_sliding', 'vergence']
        int_means = [np.mean(isc_int_values)*10, np.mean(sacc_int_values), np.mean(verg_int_values)]
        ext_means = [np.mean(isc_ext_values)*10, np.mean(sacc_ext_values), np.mean(verg_ext_values)]

        x = np.arange(len(labels))
        width = 0.35

        ax = axs[row, col] if num_rows > 1 else (axs[col] if num_cols > 1 else axs)
        rects1 = ax.bar(x - width/2, int_means, width, label='Int')
        rects2 = ax.bar(x + width/2, ext_means, width, label='Ext')

        ax.set_ylabel('Mean ET values at Ext, Int Idx')
        ax.set_title(f'{pat}_IAP')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        for j, label in enumerate(labels):
            p_value = t_test_results[label].pvalue
            ax.text(x[j], max(int_means[j], ext_means[j]), f'p={p_value:.3f}', ha='center')

# Hide any unused subplots
for idx in range(num_patients, num_rows * num_cols):
    fig.delaxes(axs.flatten()[idx])

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(fig_dir, f'IAP-ET_{region}_int_ext_avg_compare_summary.png'), bbox_inches='tight')
plt.show()


# Create a single summary figure with subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
fig.suptitle(f'Comparison of Int and Ext means for Alpha Band in all patients in {region} for {sd_mult}')

for idx, pat in enumerate(patients):
    row = idx // num_cols
    col = idx % num_cols

    pat_dir = os.path.join(data_dir, pat)
    pat_fig_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(pat_fig_dir):
        os.makedirs(pat_fig_dir)

    # Search for 'feature_df.csv' files in each patient directory
    et_files = [filename for filename in os.listdir(pat_dir) if 'ET_features_with_IAP' in filename and filename.endswith('.csv')]

    # Handle each matching file
    for et_file in et_files:
        full_et_path = os.path.join(pat_dir, et_file)
        try:
            eye_values = pd.read_csv(full_et_path)
            print("Processed eye tracking data for:", full_et_path)
        except Exception as e:
            print(f"Error reading {full_et_path}: {e}")
            continue

    # Read in the ET values
    isc = eye_values['isc_pat']
    sacc = eye_values['saccade_rate_sliding']
    verg = eye_values['vergence']

    eye_cols = {'isc_pat', 'saccade_rate_sliding', 'vergence'}

    if len(eye_values.columns) == 3:
        print("Skipping the code block as there are only 3 columns in eye_values.columns.")
    else:
        alpha_cols = [col for col in eye_values.columns if 'alpha' in col]
        # Initialize lists to collect aggregated values for each metric
        isc_int_values, isc_ext_values = [], []
        sacc_int_values, sacc_ext_values = [], []
        verg_int_values, verg_ext_values = [], []

        # Process each 'IAP' column
        for elec in alpha_cols:
            # Calculate the mean and standard deviation of the column
            col_data = eye_values[elec].values
            mean_value = col_data.mean()
            sd_value = col_data.std()

            # Create dynamic variable names for mean and standard deviation
            mean_var_name = f'mean_{elec}'
            sd_var_name = f'sd_{elec}'

            # Assign the mean and standard deviation values to dynamic variables in the globals() dictionary
            globals()[mean_var_name] = mean_value
            globals()[sd_var_name] = sd_value
            print(f'{mean_var_name} = {globals()[mean_var_name]}, {sd_var_name} = {globals()[sd_var_name]}')

            int_idx = np.where(col_data > mean_value + sd_mult*sd_value)[0]
            ext_idx = np.where(col_data <= mean_value - sd_mult*sd_value)[0]

            # Collect aggregated values for isc, sacc, and verg
            isc_int_values.extend(isc[int_idx])
            isc_ext_values.extend(isc[ext_idx])
            sacc_int_values.extend(sacc[int_idx])
            sacc_ext_values.extend(sacc[ext_idx])
            verg_int_values.extend(verg[int_idx])
            verg_ext_values.extend(verg[ext_idx])

        # Conduct a two-sample t-test comparing the aggregated values
        t_test_results = {
            'isc_pat': stats.ttest_ind(isc_int_values, isc_ext_values),
            'saccade_rate_sliding': stats.ttest_ind(sacc_int_values, sacc_ext_values),
            'vergence': stats.ttest_ind(verg_int_values, verg_ext_values)
        }

        # Generate a bar graph visualizing these comparisons
        labels = ['isc_pat', 'saccade_rate_sliding', 'vergence']
        int_means = [np.mean(isc_int_values)*10, np.mean(sacc_int_values), np.mean(verg_int_values)]
        ext_means = [np.mean(isc_ext_values)*10, np.mean(sacc_ext_values), np.mean(verg_ext_values)]

        x = np.arange(len(labels))
        width = 0.35

        ax = axs[row, col] if num_rows > 1 else (axs[col] if num_cols > 1 else axs)
        rects1 = ax.bar(x - width/2, int_means, width, label='Int')
        rects2 = ax.bar(x + width/2, ext_means, width, label='Ext')

        ax.set_ylabel('Mean ET values at ext, int IDX')
        ax.set_title(f'{pat} 8-13 Hz')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        for j, label in enumerate(labels):
            p_value = t_test_results[label].pvalue
            ax.text(x[j], max(int_means[j], ext_means[j]), f'p={p_value:.3f}', ha='center')

# Hide any unused subplots
for idx in range(num_patients, num_rows * num_cols):
    fig.delaxes(axs.flatten()[idx])

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(fig_dir, f'Alpha-ET_{region}_int_ext_avg_compare_summary.png'), bbox_inches='tight')
plt.show()

int_means = []
ext_means = []
        
for pat in patients:
    pat_dir = os.path.join(data_dir, pat)
    pat_fig_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(pat_fig_dir):
        os.makedirs(pat_fig_dir)

    # Search for 'feature_df.csv' files in each patient directory
    et_files = [filename for filename in os.listdir(pat_dir) if 'ET_features_with_IAP' in filename and filename.endswith('.csv')]

    # Handle each matching file
    for et_file in et_files:
        full_et_path = os.path.join(pat_dir, et_file)
        try:
            eye_values = pd.read_csv(full_et_path)
            print("Processed eye tracking data for:", full_et_path)
        except Exception as e:
            print(f"Error reading {full_et_path}: {e}")
            continue
    
    # Read in the ET values
    isc = eye_values['isc_pat']
    sacc = eye_values['saccade_rate_sliding']
    verg = eye_values['vergence']
 
    eye_cols = {'isc_pat', 'saccade_rate_sliding', 'vergence'}
    
    if len(eye_values.columns) == 3:
        print("Skipping the code block as there are only 3 columns in eye_values.columns.")
    else:
        IAP_cols = [col for col in eye_values.columns if 'IAP' in col]
        
        # Initialize lists to collect aggregated values for each metric
        isc_int_values, isc_ext_values = [], []
        sacc_int_values, sacc_ext_values = [], []
        verg_int_values, verg_ext_values = [], []
        
        # Process each 'IAP' column
        for elec in IAP_cols:
            # Calculate the mean and standard deviation of the column
            col_data = eye_values[elec].values
            mean_value = col_data.mean()
            sd_value = col_data.std()
            
            # Create dynamic variable names for mean and standard deviation
            mean_var_name = f'mean_{elec}'
            sd_var_name = f'sd_{elec}'
            
            # Assign the mean and standard deviation values to dynamic variables in the globals() dictionary
            globals()[mean_var_name] = mean_value
            globals()[sd_var_name] = sd_value
            print(f'{mean_var_name} = {globals()[mean_var_name]}, {sd_var_name} = {globals()[sd_var_name]}')
    
            int_idx = np.where(col_data > mean_value + 1*sd_value)[0]
            ext_idx = np.where(col_data <= mean_value - 1*sd_value)[0]
    
            # Collect aggregated values for isc, sacc, and verg
            isc_int_values.extend(isc[int_idx])
            isc_ext_values.extend(isc[ext_idx])
            sacc_int_values.extend(sacc[int_idx])
            sacc_ext_values.extend(sacc[ext_idx])
            verg_int_values.extend(verg[int_idx])
            verg_ext_values.extend(verg[ext_idx])
    
        # Conduct a two-sample t-test comparing the aggregated values
        t_test_results = {
            'isc_pat': stats.ttest_ind(isc_int_values, isc_ext_values),
            'saccade_rate_sliding': stats.ttest_ind(sacc_int_values, sacc_ext_values),
            'vergence': stats.ttest_ind(verg_int_values, verg_ext_values)
        }
    
        # Generate a bar graph visualizing these comparisons
        labels = ['isc_pat', 'saccade_rate_sliding', 'vergence']
        int_means = [np.mean(isc_int_values), np.mean(sacc_int_values), np.mean(verg_int_values)]
        ext_means = [np.mean(isc_ext_values), np.mean(sacc_ext_values), np.mean(verg_ext_values)]
    
        x = np.arange(len(labels))
        width = 0.35
    
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, int_means, width, label='Int')
        rects2 = ax.bar(x + width/2, ext_means, width, label='Ext')
    
        ax.set_ylabel('mean of Values')
        ax.set_title(f'Comparison of Int and Ext means for {pat} in {region}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
    
        for i, label in enumerate(labels):
            p_value = t_test_results[label].pvalue
            ax.text(x[i], max(int_means[i], ext_means[i]), f'p={p_value:.3f}', ha='center')
    
        fig.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'IAF-ET_{region}_{pat}_int_ext_avg_compare.png'), bbox_inches='tight')
        plt.show()


#%%


# Define a function to add mean and SD lines
def add_mean_sd_lines(data, ax):
    mean_val = np.mean(data)
    std_val = np.std(data)
    ax.axhline(mean_val, color='green', linestyle='--')
    ax.axhline(mean_val + std_val, color='blue', linestyle=':')
    ax.axhline(mean_val - std_val, color='orange', linestyle=':')
    
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def add_mean_sd_lines(data, ax):
    mean = np.mean(data)
    std = np.std(data)
    ax.axhline(mean, color='blue', linestyle='--', linewidth=2, label='Mean')
    ax.axhline(mean + std, color='green', linestyle='--', linewidth=2, label='+1 SD')
    ax.axhline(mean - std, color='green', linestyle='--', linewidth=2, label='-1 SD')
    #%%
 # Plot ET values with potential internal and external attention indices
 # Plot ISC
 plt.figure(figsize=(14, 10))
 plt.title(f'Possible Attention State Indices measures for {pat} in {region}', fontsize=16)
 plt.subplot(3, 1, 1)
 plt.plot(isc,color = "black")
 #plt.scatter(contrast_indices, isc[contrast_indices], color='red', label='Contrast Points', zorder=5)
 plt.scatter(int_idx, isc[int_idx], color='yellow', label='ind_idx', zorder=5)
 plt.scatter(ext_idx, isc[ext_idx], color='red', label='ext_idx', zorder=5)
 add_mean_sd_lines(isc, plt.gca())
 plt.title('ISC')
 plt.xlabel('Time')
 plt.ylabel('ISC')
 plt.legend()
 
 # Plot Saccade Rate
 plt.subplot(3, 1, 2)
 plt.plot(sacc,color = "black")
 #plt.scatter(contrast_indices, sacc[contrast_indices], color='red', label='Contrast Points', zorder=5)
 plt.scatter(int_idx, sacc[int_idx], color='yellow', label='ind_idx', zorder=5)
 plt.scatter(ext_idx, sacc[ext_idx], color='red', label='ext_idx', zorder=5)
 add_mean_sd_lines(sacc, plt.gca())
 plt.title('Saccade Rate')
 plt.xlabel('Time')
 plt.ylabel('Saccade Rate')
 plt.legend()
 
 # Plot Vergence
 plt.subplot(3, 1, 3)
 plt.plot(verg,color = "black")
 #plt.scatter(contrast_indices, verg[contrast_indices], color='red', label='Contrast Points', zorder=5)=
 plt.scatter(int_idx, verg[int_idx], color='yellow', label='ind_idx', zorder=5)
 plt.scatter(ext_idx, verg[ext_idx], color='red', label='ext_idx', zorder=5)
 add_mean_sd_lines(verg, plt.gca())
 plt.title('Gaze Vergence')
 plt.xlabel('Time')
 plt.ylabel('Vergence')
 plt.legend()
 plt.tight_layout()
 
#%%
patients = [entry for entry in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, entry))]
patients.sort()
if 'NS135' in patients:
    patients.remove('NS135')
if 'NS136' in patients:
    patients.remove('NS136')
if 'NS153' in patients:
    patients.remove('NS153')

# Determine the number of rows needed for 3 columns
num_patients = len(patients)
num_cols = 3
num_rows = (num_patients + num_cols - 1) // num_cols

# Create a single summary figure with subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
fig.suptitle(f'Comparison of Int and Ext means for Peak Alpha all patients in {region}')

for idx, pat in enumerate(patients):
    row = idx // num_cols
    col = idx % num_cols

    pat_dir = os.path.join(data_dir, pat)
    pat_fig_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(pat_fig_dir):
        os.makedirs(pat_fig_dir)

    # Search for 'feature_df.csv' files in each patient directory
    et_files = [filename for filename in os.listdir(pat_dir) if 'ET_features_with_IAP' in filename and filename.endswith('.csv')]

    # Handle each matching file
    for et_file in et_files:
        full_et_path = os.path.join(pat_dir, et_file)
        try:
            eye_values = pd.read_csv(full_et_path)
            print("Processed eye tracking data for:", full_et_path)
        except Exception as e:
            print(f"Error reading {full_et_path}: {e}")
            continue

    # Read in the ET values
    isc = eye_values['isc_pat']
    sacc = eye_values['saccade_rate_sliding']
    verg = eye_values['vergence']

    eye_cols = {'isc_pat', 'saccade_rate_sliding', 'vergence'}

    if len(eye_values.columns) == 3:
        print("Skipping the code block as there are only 3 columns in eye_values.columns.")
    else:
        IAP_cols = [col for col in eye_values.columns if 'IAP' in col]
        # Initialize lists to collect aggregated values for each metric
        isc_int_values, isc_ext_values = [], []
        sacc_int_values, sacc_ext_values = [], []
        verg_int_values, verg_ext_values = [], []

        # Process each 'IAP' column
        for elec in IAP_cols:
            # Calculate the mean and standard deviation of the column
            col_data = eye_values[elec].values
            mean_value = col_data.mean()
            sd_value = col_data.std()

            # Create dynamic variable names for mean and standard deviation
            mean_var_name = f'mean_{elec}'
            sd_var_name = f'sd_{elec}'

            # Assign the mean and standard deviation values to dynamic variables in the globals() dictionary
            globals()[mean_var_name] = mean_value
            globals()[sd_var_name] = sd_value
            print(f'{mean_var_name} = {globals()[mean_var_name]}, {sd_var_name} = {globals()[sd_var_name]}')

            int_idx = np.where(col_data > mean_value + 1*sd_value)[0]
            ext_idx = np.where(col_data <= mean_value - 1*sd_value)[0]

            # Collect aggregated values for isc, sacc, and verg
            isc_int_values.extend(isc[int_idx])
            isc_ext_values.extend(isc[ext_idx])
            sacc_int_values.extend(sacc[int_idx])
            sacc_ext_values.extend(sacc[ext_idx])
            verg_int_values.extend(verg[int_idx])
            verg_ext_values.extend(verg[ext_idx])

        # Conduct a two-sample t-test comparing the aggregated values
        t_test_results = {
            'isc_pat': stats.ttest_ind(isc_int_values, isc_ext_values),
            'saccade_rate_sliding': stats.ttest_ind(sacc_int_values, sacc_ext_values),
            'vergence': stats.ttest_ind(verg_int_values, verg_ext_values)
        }

        # Generate a bar graph visualizing these comparisons
        labels = ['isc_pat', 'saccade_rate_sliding', 'vergence']
        int_means = [np.mean(isc_int_values)*10, np.mean(sacc_int_values), np.mean(verg_int_values)]
        ext_means = [np.mean(isc_ext_values)*10, np.mean(sacc_ext_values), np.mean(verg_ext_values)]

        x = np.arange(len(labels))
        width = 0.35

        ax = axs[row, col] if num_rows > 1 else (axs[col] if num_cols > 1 else axs)
        rects1 = ax.bar(x - width/2, int_means, width, label='Int')
        rects2 = ax.bar(x + width/2, ext_means, width, label='Ext')

        ax.set_ylabel('Mean ET values at Ext, Int Idx')
        ax.set_title(f'{pat}_IAP')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        for j, label in enumerate(labels):
            p_value = t_test_results[label].pvalue
            ax.text(x[j], max(int_means[j], ext_means[j]), f'p={p_value:.3f}', ha='center')

# Hide any unused subplots
for idx in range(num_patients, num_rows * num_cols):
    fig.delaxes(axs.flatten()[idx])

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(fig_dir, f'IAP-ET_{region}_int_ext_avg_compare_summary.png'), bbox_inches='tight')
plt.show()


# Create a single summary figure with subplots
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
fig.suptitle(f'Comparison of Int and Ext means for Alpha Band in all patients in {region}')

for idx, pat in enumerate(patients):
    row = idx // num_cols
    col = idx % num_cols

    pat_dir = os.path.join(data_dir, pat)
    pat_fig_dir = os.path.join(fig_dir, pat)
    if not os.path.exists(pat_fig_dir):
        os.makedirs(pat_fig_dir)

    # Search for 'feature_df.csv' files in each patient directory
    et_files = [filename for filename in os.listdir(pat_dir) if 'ET_features_with_IAP' in filename and filename.endswith('.csv')]

    # Handle each matching file
    for et_file in et_files:
        full_et_path = os.path.join(pat_dir, et_file)
        try:
            eye_values = pd.read_csv(full_et_path)
            print("Processed eye tracking data for:", full_et_path)
        except Exception as e:
            print(f"Error reading {full_et_path}: {e}")
            continue

    # Read in the ET values
    isc = eye_values['isc_pat']
    sacc = eye_values['saccade_rate_sliding']
    verg = eye_values['vergence']

    eye_cols = {'isc_pat', 'saccade_rate_sliding', 'vergence'}

    if len(eye_values.columns) == 3:
        print("Skipping the code block as there are only 3 columns in eye_values.columns.")
    else:
        alpha_cols = [col for col in eye_values.columns if 'alpha' in col]
        # Initialize lists to collect aggregated values for each metric
        isc_int_values, isc_ext_values = [], []
        sacc_int_values, sacc_ext_values = [], []
        verg_int_values, verg_ext_values = [], []

        # Process each 'IAP' column
        for elec in alpha_cols:
            # Calculate the mean and standard deviation of the column
            col_data = eye_values[elec].values
            mean_value = col_data.mean()
            sd_value = col_data.std()

            # Create dynamic variable names for mean and standard deviation
            mean_var_name = f'mean_{elec}'
            sd_var_name = f'sd_{elec}'

            # Assign the mean and standard deviation values to dynamic variables in the globals() dictionary
            globals()[mean_var_name] = mean_value
            globals()[sd_var_name] = sd_value
            print(f'{mean_var_name} = {globals()[mean_var_name]}, {sd_var_name} = {globals()[sd_var_name]}')

            int_idx = np.where(col_data > mean_value + 1*sd_value)[0]
            ext_idx = np.where(col_data <= mean_value - 1*sd_value)[0]

            # Collect aggregated values for isc, sacc, and verg
            isc_int_values.extend(isc[int_idx])
            isc_ext_values.extend(isc[ext_idx])
            sacc_int_values.extend(sacc[int_idx])
            sacc_ext_values.extend(sacc[ext_idx])
            verg_int_values.extend(verg[int_idx])
            verg_ext_values.extend(verg[ext_idx])

        # Conduct a two-sample t-test comparing the aggregated values
        t_test_results = {
            'isc_pat': stats.ttest_ind(isc_int_values, isc_ext_values),
            'saccade_rate_sliding': stats.ttest_ind(sacc_int_values, sacc_ext_values),
            'vergence': stats.ttest_ind(verg_int_values, verg_ext_values)
        }

        # Generate a bar graph visualizing these comparisons
        labels = ['isc_pat', 'saccade_rate_sliding', 'vergence']
        int_means = [np.mean(isc_int_values)*10, np.mean(sacc_int_values), np.mean(verg_int_values)]
        ext_means = [np.mean(isc_ext_values)*10, np.mean(sacc_ext_values), np.mean(verg_ext_values)]

        x = np.arange(len(labels))
        width = 0.35

        ax = axs[row, col] if num_rows > 1 else (axs[col] if num_cols > 1 else axs)
        rects1 = ax.bar(x - width/2, int_means, width, label='Int')
        rects2 = ax.bar(x + width/2, ext_means, width, label='Ext')

        ax.set_ylabel('Mean of Eye Feature Values at Ext or Int Indices')
        ax.set_title(f'{pat} 7-13 Hz')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        for j, label in enumerate(labels):
            p_value = t_test_results[label].pvalue
            ax.text(x[j], max(int_means[j], ext_means[j]), f'p={p_value:.3f}', ha='center')

# Hide any unused subplots
for idx in range(num_patients, num_rows * num_cols):
    fig.delaxes(axs.flatten()[idx])

fig.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(fig_dir, f'Alpha-ET_{region}_int_ext_avg_compare_summary.png'), bbox_inches='tight')
plt.show()

