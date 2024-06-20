#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:09:29 2024

@author: christine
"""

Script for calculating and plotting the differences in alpha power based on indices 
derived from eye features (isc, vergence, saccade rate)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:06:23 2024

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:33:15 2024

@author: christinechesebrough
"""

from scipy.stats import pearsonr
import os
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
region = 'middle frontal' 

conditions = ['isc_verg_sacc', 'isc_verg', 'isc_sacc', 'sacc_verg', 'isc', 'verg', 'sacc']

data_dir = '/Volumes/Expansion/Movie_data/ET_features/mid_front/derived_data'
fig_dir = os.path.join(f'/Volumes/Expansion/Movie_data/old_verg/{region}/')
    
# Ensure the fig_dir exists
os.makedirs(fig_dir, exist_ok=True)

# Initialize a list to store subplot data
subplot_data = []

for condition in conditions:
    print(f"Data directory: {data_dir}")
    print(f"Figure directory: {fig_dir}")
    
    patients = [entry for entry in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, entry))]
    patients.sort()
    
    all_int_avgs = []
    all_ext_avgs = []
    patient_labels = []
    significance_markers = []

    for pat in patients:
        pat_dir = os.path.join(data_dir, pat)
        pat_fig_dir = os.path.join(fig_dir, pat)
        if not os.path.exists(pat_fig_dir):
            os.makedirs(pat_fig_dir)
    
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
                continue
        
        # Read in the ET values
        isc = eye_values['isc_pat']
        sacc = eye_values['saccade_rate_sliding']
        verg = eye_values['vergence']
        
        # Find the mean and std of each of the ET values
        mean_isc, std_isc = np.mean(isc), np.std(isc)
        mean_sacc, std_sacc = np.mean(sacc), np.std(sacc)
        mean_verg, std_verg = np.mean(verg), np.std(verg)
    
        if condition == 'isc_verg_sacc':
            # Calculate int_idx: Indices where isc < mean, sacc < mean, and verg > mean
            int_idx = np.where((isc < mean_isc) & (sacc < mean_sacc) & (verg > mean_verg))[0]
            
            # Calculate ext_idx: Indices where isc > mean, sacc > mean, and verg < mean
            ext_idx = np.where((isc > mean_isc) & (sacc > mean_sacc) & (verg < mean_verg))[0]
       
        elif condition == 'isc_verg':
            # Calculate int_idx: Indices where isc < mean, and verg > mean
            int_idx = np.where((isc < mean_isc) & (verg > mean_verg))[0]
            
            # Calculate ext_idx: Indices where isc > mean, and verg < mean
            ext_idx = np.where((isc > mean_isc) & (verg < mean_verg))[0]
            
        elif condition == 'isc_sacc':
            # Calculate int_idx: Indices where isc < mean, sacc < mean
            int_idx = np.where((isc < mean_isc) & (sacc < mean_sacc))[0]
            
            # Calculate ext_idx: Indices where isc > mean, sacc > mean
            ext_idx = np.where((isc > mean_isc) & (sacc > mean_sacc))[0]
          
        elif condition == 'sacc_verg':
            # Calculate int_idx: Indices where sacc < mean, and verg > mean
            int_idx = np.where((sacc < mean_sacc) & (verg > mean_verg))[0]
            
            # Calculate ext_idx: Indices where sacc > mean, and verg < mean
            ext_idx = np.where((sacc > mean_sacc) & (verg < mean_verg))[0]
        
        elif condition == 'sacc':
            # Calculate int_idx: Indices where sacc < mean
            int_idx = np.where(sacc < mean_sacc)[0]
            
            # Calculate ext_idx: Indices where sacc > mean
            ext_idx = np.where(sacc > mean_sacc)[0]
        
        elif condition == 'isc':
            # Calculate int_idx: Indices where isc < mean
            int_idx = np.where(isc < mean_isc)[0]
            
            # Calculate ext_idx: Indices where isc > mean
            ext_idx = np.where(isc > mean_isc)[0]
            
        elif condition == 'verg':
            # Calculate int_idx: Indices where verg > mean
            int_idx = np.where(verg > mean_verg)[0]
            
            # Calculate ext_idx: Indices where verg < mean
            ext_idx = np.where(verg < mean_verg)[0]
        
        else:
            raise ValueError(f"Unknown condition: {condition}")
    
        
        # Continue with further processing using int_idx and ext_idx
        print(f"Patient: {pat}")
        print(f"Int indices: {int_idx}")
        print(f"Ext indices: {ext_idx}")
        
        # Extract neuro data and compare between periods of int_idx and ext_idx
        eye_cols = {'isc_pat', 'saccade_rate_sliding', 'vergence'}
            
        if len(eye_values.columns) == 3:
            print("Skipping the code block as there are only 3 columns in eye_values.columns.")
        else:
            aggregated_int_values = []
            aggregated_ext_values = []
    
            for elec in eye_values.columns:
                if elec not in eye_cols:
                    col_data = eye_values[elec].values
    
                    int_vals = col_data[int_idx]
                    ext_vals = col_data[ext_idx]
    
                    aggregated_int_values.extend(int_vals)
                    aggregated_ext_values.extend(ext_vals)
    
            int_avg = np.mean(aggregated_int_values) if len(aggregated_int_values) > 0 else np.nan
            ext_avg = np.mean(aggregated_ext_values) if len(aggregated_ext_values) > 0 else np.nan
    
            all_int_avgs.append(int_avg)
            all_ext_avgs.append(ext_avg)
            patient_labels.append(pat)
    
            # Perform t-test between int and ext values
            if len(aggregated_int_values) > 0 and len(aggregated_ext_values) > 0:
                t_stat, p_val = ttest_ind(aggregated_int_values, aggregated_ext_values, nan_policy='omit')
                if p_val < 0.05:
                    significance_markers.append(True)
                else:
                    significance_markers.append(False)
            else:
                significance_markers.append(False)
    
       
    # Store subplot data for summary figure
    subplot_data.append((all_int_avgs, all_ext_avgs, patient_labels, significance_markers, condition))

# Create a summary figure with subplots for each condition
num_conditions = len(conditions)
fig, axs = plt.subplots(2, 4, figsize=(28, 20))  # Adjusted to fit 7 subplots in 2 rows
fig.suptitle('Internal vs External Attention Averages for all Conditions', fontsize=24)

for i, (int_avgs, ext_avgs, labels, markers, cond) in enumerate(subplot_data):
    row = i // 4
    col = i % 4
    ax = axs[row, col] if i < 7 else axs[1, 3]  # Ensure all subplots are placed correctly
    
    bar_width = 0.35
    index = np.arange(len(labels))
    
    ax.bar(index, int_avgs, bar_width, label='Int Avg', color='blue')
    ax.bar(index + bar_width, ext_avgs, bar_width, label='Ext Avg', color='orange')
    
    for j in range(len(labels)):
        if markers[j]:
            max_avg = max(int_avgs[j], ext_avgs[j])
            ax.text(j + bar_width / 2, max_avg, '*', ha='center', va='bottom', fontsize=14)
    
    ax.set_xlabel('Patients', fontsize=14)
    ax.set_ylabel('Standardized Power (8-12 Hz)', fontsize=14)
    ax.set_title(f'{cond} features in {region}', fontsize=16)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.legend(fontsize=12)

# Remove the empty subplot
fig.delaxes(axs[1, 3])

plt.tight_layout(rect=[0, 0, 1, 0.95])
summary_plot_path = os.path.join(f'/Volumes/Expansion/Movie_data/old_verg/{region}', '{region}_summary_int_ext_avg_compare.png')
plt.savefig(summary_plot_path, bbox_inches='tight')
plt.show()



