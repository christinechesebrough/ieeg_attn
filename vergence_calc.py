#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:21:58 2024

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:39:52 2024

@author: christinechesebrough
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:19:08 2024

@author: christinechesebrough
"""

# Calculate differences in gaze positions between right and left eye (gaze disparity, disparity of visual angle)

import os, sys, re
import numpy as np
import mne
from itertools import compress
from pynwb import NWBHDF5IO
from helpers import interp_bad_samples, combine_left_right, detect_saccades_remodnav
from tqdm import tqdm
from scipy import ndimage
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import scipy.interpolate as interp
import copy, math
import scipy.interpolate as interp
import scipy.signal as signal
import scipy.stats as stats
import remodnav

sys.path.append('/Users/christinechesebrough/Documents/EPIPE-master/Python')
from epipe import inspectNwb, nwb2mne, read_ielvis, reref_avg, reref_bipolar, filter_hfa_continuous

 #%%

def interp_bad_data(data_gaze, data_val, k_small, k_large, k_delay, visualize):
   
    """
    Interpolate bad samples in gaze data
    
    Inputs:
        data_gaze       - data array [samples x dimension (xy)]
        data_val        - index of samples with invalid data [samples]
        k_small         - number of samples of small gaps that are filled
        k_large         - number of samples to add to large gaps (blinks, etc)
        k_delay         - number of samples to shift delay around large gaps
        visualize       - flag to plot gaze data before and after processing
        
    Returns:
        data_gaze       - processed data vector
        data_val        - upadted vector of invalid data
    """
    
    data_val = copy.copy(data_val)
    
    # Plot signal
    if visualize:
        plt.figure()
        plt.plot(data_gaze)
    
    # Create sample vector
    samples = np.arange(0, len(data_gaze))
    
    # Interpolate small gaps
    idx_gap = signal.convolve(data_val, np.ones(k_small), 'same')
    gap_val = signal.convolve(idx_gap==0, np.ones((k_small)), mode='same') == 0
    
    gaps = np.logical_and(gap_val, np.invert(data_val))
    data_val[gaps] = True
    
    f_gaze = interp.interp1d(samples[np.invert(gaps)],
                             data_gaze[np.invert(gaps)], 
                             kind='linear', fill_value="extrapolate")
    data_gaze[gaps] = f_gaze(samples[gaps])
    
    # Remove samples around longer gaps (mostly blinks)
    idx_bad = signal.convolve(np.invert(data_val), 
                              np.concatenate([np.zeros(k_delay),
                                              np.ones(k_large)]), 
                              'same') > 1
    
    data_gaze[idx_bad] = np.nan
    data_val[idx_bad] = False
    
    if visualize:
        plt.plot(data_gaze)
        
    return data_gaze, data_val


#%% Eye tracking parameters

# Frequency range
freq_range = [70, 170]
n_freq_bins = 10
freq_space = 'log'      # 'log', 'lin'

resample_bha_fs = 100
t_diff = 0.1

dist = 60

# Screen dimentions [cm]
d = 23.8*2.54
ar = 16/9

k_small = 15 #15
k_large = 90 #90
k_delay = 20
vis_bads = False

screen_pix = np.array([1920, 1080])
screen_cm = np.array([50.92, 28.64])
    
#%%
data_dir = '/Volumes/Expansion/Movie_data/movies_nwb_standard'
fs_dir = '/Volumes/Documents/Movie_data/anatomy'
eloc_dir = '/Volumes/Expansion/Movie_data/data/electrode_localization'
prep_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'

et_prep_dir = 'Eye_prep'
audio_dir = 'Audio'
neural_prep_dir = 'Neural_prep'
hfa_dir = 'HFA'

#%%  
broken_nwb = [] 

patients = [patient for patient in os.listdir(data_dir) if not patient.startswith('.DS_Store')]
patients.sort()

for pat in patients:
    directory_path = os.path.join(data_dir, pat)
    all_files = os.listdir(directory_path)
    
    implants = [filename for filename in all_files if os.path.isdir(os.path.join(directory_path, filename)) and not filename == '.DS_Store']
    
    for imp in implants:
        
        num_imp = int(re.findall(r'\d+', imp)[0])
        
        if num_imp == 1:
            pat_fs = pat.replace('sub-', '')
        elif num_imp >= 2:
            pat_fs = f'{pat.replace("sub-", "")}_{num_imp:02d}'
        
            # Pupil data
        sub_et_prep_dir = '{:s}/{:s}/{:s}'.format(prep_dir, pat_fs, et_prep_dir)
            
        if not os.path.exists(sub_et_prep_dir):
            os.makedirs(sub_et_prep_dir)

        movies = os.listdir('{:s}/{:s}/{:s}'.format(data_dir, pat, imp))    
                
        # Filter movies to include only files with the string "task-despicable_me_english"
        filtered_movies = [mov for mov in movies if "task-despicable_me_english" in mov]
        
        # If no such movie is found, you may want to handle it (e.g., raise an error or continue)
        if not filtered_movies:
            raise FileNotFoundError("No movie file with 'task-despicable_me_english' found in the directory.")
        
        # Process the filtered movies
        for mov in filtered_movies:
            # Construct the file path in a safe manner
            et_prep_filename = '{:s}/{:s}'.format(sub_et_prep_dir, mov.replace('_ieeg.nwb', '_et_prep.csv'))
        
            nwb_fname = os.path.join(data_dir, pat, imp, mov)  
            
            # NWB read       
            io = NWBHDF5IO(nwb_fname, mode='r', load_namespaces=True)
            nwb = io.read()
            print('Loading nwb data for patient {:s} ...'.format(pat))
            
            #  Preprocess and visualize eyetracking data

             # Get position and time
             #adcs is gaze position on screen 2D vector x, y between 0 and 1
            l_gaze = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].data[:]
            r_gaze = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].data[:]
             
            x_right = r_gaze[:,0]
            y_right = r_gaze[:,1]
             
            x_left = l_gaze[:,0]
            y_left = l_gaze[:,1]
            
            x_right_orig = x_right
            y_right_orig = y_right
            x_left_orig = x_left
            y_left_orig = y_left
            
            # a, b, c, points in space tracking eye location in three dimensions    
            eye_pos_left = nwb.processing['eye_tracking']['eyes']['l_eye_pos'].data[:]
            eye_pos_right = nwb.processing['eye_tracking']['eyes']['r_eye_pos'].data[:]
            
            # same point as gaze position on screen but measured from origin at the eye tracker
            gaze_pos_left = nwb.processing['eye_tracking']['eyes']['l_eye_gaze'].data[:]
            gaze_pos_right = nwb.processing['eye_tracking']['eyes']['r_eye_gaze'].data[:]
            
            # Calculate the approximate interpupillary distance 
            
            # Calculate the Euclidean distance between corresponding pairs of eye position points
            distances = np.sqrt(np.sum((eye_pos_left - eye_pos_right) ** 2, axis=1))
            
            # The IPD can be approximated as the median of the Euclidean distance between the two eye positions in space.
            IPD_cm = np.median(distances) / 10  # If distances are in mm, convert to cm.
            
            # time
            t_nwb = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].timestamps[:]
            
            # Validity
            val_left = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:] <= 1
            val_right = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:] <= 1
            comparison_array = np.where(val_left == val_right, 0, 1)

            #plt.plot(t_nwb,x_right,c='red',label = 'horz_right')
            #plt.plot(t_nwb,x_left,c='blue',label = 'horz_left')
            
            val_left_orig = nwb.processing['eye_tracking']['eyes']['l_eye_adcs'].control[:]
            val_right_orig = nwb.processing['eye_tracking']['eyes']['r_eye_adcs'].control[:]
            
            # Sampling rate
            fs_eye = 1/np.median(np.diff(t_nwb))

            # Exclude data outside the screen 
            val_left[np.logical_or(y_left > 1, y_left < 0)] = False   
            val_left[np.logical_or(x_left > 1, x_left < 0)] = False
            val_right[np.logical_or(y_right > 1, y_right < 0)] = False
            val_right[np.logical_or(x_right > 1, x_right < 0)] = False
        
            # Interpolate bad data for both eyes
            x_left, val_left_x = interp_bad_data(x_left, val_left,k_small, k_large, k_delay, vis_bads)
       #     y_left, val_left_y = interp_bad_data(y_left, val_left,,k_small, k_large, k_delay, vis_bads)
            x_right, val_right_x = interp_bad_data(x_right, val_right,k_small, k_large, k_delay, vis_bads)
        #    y_right, val_right_y = interp_bad_data(y_right, val_right, k_small, k_large, k_delay, vis_bads)

            
            # Calculate visual disparity before interpolation
            x_diff = x_right - x_left
        
            # Plot results
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, x_diff, label='x_disparity (not interpolated)')
            plt.xlabel('Time')
            plt.ylabel('degrees of visual angle')
            plt.title('x_disparity (not interpolated)')
            ax.legend()
            ax.set_title(f'Patient {pat} z gaze disparity (DVA)')
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}_x_dva_diff.png")
            fig.savefig(fig_filename)
            plt.close(fig)  
            
            # Interpolate over disparity values
            interp_kind='linear'
            
            # Mark bad data as NaN for x and y coordinates
            x_left[~val_left] = np.nan
            x_right[~val_right] = np.nan
            
            # Calculate visual disparity before interpolation
            x_diff = x_right - x_left
            
            # Create a boolean array indicating where valid data is present in both eyes for x 
            valid_x = ~np.isnan(x_diff)
            
            # Create an array of sample indices
            eye_samples_x = np.arange(len(x_diff))
            
            # Interpolate over gaps in x_diff and y_diff
            f_x_diff = interp.interp1d(eye_samples_x[valid_x], x_diff[valid_x], kind=interp_kind, fill_value='extrapolate')
            x_diff_filled = f_x_diff(eye_samples_x)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, x_diff_filled, c="orange", label='x gaze distance interpolated')
            plt.xlabel('Time')
            plt.ylabel('degrees of visual angle')
            plt.title('x gaze disparity (DVA) (interpolated)')
            ax.legend()
            ax.set_title(f'Patient {pat} x gaze disparity (DVA) (interpolated)')
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}_x_dva_diff_interp.png")
            fig.savefig(fig_filename)
            plt.close(fig)  
            
            # Calculate gaze disparity in terms of pixel distance
            
            eye_pos_left_interp, _ = interp_bad_data(eye_pos_left[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
            eye_pos_right_interp, _ = interp_bad_data(eye_pos_right[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
            gaze_pos_left_interp, _ = interp_bad_data(gaze_pos_left[:, 2], val_left, k_small, k_large, k_delay, vis_bads)
            gaze_pos_right_interp, _ = interp_bad_data(gaze_pos_right[:, 2], val_left, k_small, k_large, k_delay, vis_bads)

            dist_left = eye_pos_left_interp - gaze_pos_left_interp
            dist_right = eye_pos_right_interp - gaze_pos_right_interp

            # Distance in cm (converted from median distance in mm to cm for both eyes)
            dist_left_cm = dist_left / 10
            dist_right_cm = dist_right / 10

            #avg distance should be the median from both eyes averaged? Maybe this shouldn't be a constant
            avg_dist = (dist_left_cm + dist_right_cm)/2
             
            #converting gaze position to pixel dimensions 
            
            #left eye
            x_left = x_left* screen_pix[0]  # Apply x dimension
            #x_left = x_left_filled * screen_pix[0]  # Apply x dimension
            n_left = len(x_left)
            
            data_left = np.core.records.fromarrays([x_left],names=['x'])
            
            # Right eye
            x_right = x_right * screen_pix[0]  # Apply x dimension
            #x_right = x_right_filled * screen_pix[0]  # Apply x dimension
            n_right = len(x_right)
            
            data_right = np.core.records.fromarrays([x_right],names=['x'])
            
            # Calculate px2deg for each dimension using the specific width and height in cm
            px2deg_x_left = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_left_cm))) / screen_pix[0]
            px2deg_x_right = np.rad2deg(2 * np.arctan(screen_cm[0] / (2 * dist_right_cm))) / screen_pix[0]
            
            # Apply dva calculation to x and y coordinates for both eyes
            data_left['x'] *= px2deg_x_left
            data_right['x'] *= px2deg_x_right
            
            n = len(data_left)  # or len(data_right), since they should be the same
             
            # Calculate disparity in angle of eye vergence for
            gaze_disparity_x = data_right['x'] - data_left['x']
            
            #interpolate gaze disparity in DVA
            # Create a boolean array indicating where valid data is present in both eyes for x and y
            valid_disp_x = ~np.isnan(gaze_disparity_x)
            
            # Create an array of sample indices
            eye_samples_x = np.arange(len(gaze_disparity_x))

            # Interpolate over gaps in x_diff 
            f_x_disp = interp.interp1d(eye_samples_x[valid_disp_x], gaze_disparity_x[valid_disp_x], kind=interp_kind, fill_value='extrapolate')
            x_disp_filled = f_x_disp(eye_samples_x)

          # Plot results
            plt.figure(figsize=(14, 6))
            plt.plot(t_nwb, x_disp_filled, c="blue", label='x gaze disp (DVA) interpolated')
            plt.xlabel('Time')
            plt.ylabel('X Disparity')
            plt.legend()
            plt.tight_layout()
           # plt.show()
            plt.close()  
                          
            #Calculate visual focus displacement per Huang et al. (2019)    
            
            beta = 0.283
            PD_cm = IPD_cm  # Use the approximated IPD from before
            
            # Calculate E (gaze disparity in the world coordinate)
            G = np.linalg.norm(gaze_disparity_x)  # Euclidean norm of the gaze disparity vector
            E = beta * G
            
            # Convert D from cm to mm for consistency with formula
            D_mm = ((dist_left_cm + dist_right_cm) / 2) * 10  # Average distance from eyes to screen converted to mm
                
            # Initialize visual focus displacement array
            visual_focus_displacement = np.zeros(len(gaze_disparity_x))
                 
            # Use the actual distance in the visual focus displacement calculation
            for i in range(len(gaze_disparity_x)):
                G = np.linalg.norm(gaze_disparity_x[i])  # G is euclidean norm (2) of gaze disparity
                E = beta * G  # Calculate E using the current G
            
                # Apply conditions for convergence and divergence
                if gaze_disparity_x[i] > 0:  # Divergence
                    visual_focus_displacement[i] = E * (dist_left_cm[i] + dist_right_cm[i]) * 5 / (PD_cm * 10 - E)
                else:  # Convergence
                    visual_focus_displacement[i] = -E * (dist_left_cm[i] + dist_right_cm[i]) * 5 / (PD_cm * 10 + E)


              # Generate figure of vis focus disparity before interpolation to save
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, visual_focus_displacement, c = 'pink',label='Visual Focus Displacement')
            plt.xlabel('Time')
            plt.ylabel('Displacement (mm)')
            plt.title('Visual Focus Displacement Over Time')
            ax.legend()
            ax.set_title(f'Patient {pat} Vis Focus Disparity')
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}_vf_disp_plot.png")
            fig.savefig(fig_filename)
            plt.close(fig) 
            
            # Assuming visual_focus_displacement and interp_kind are defined
           # valid_vis_fd = ~np.isnan(visual_focus_displacement)
           # eye_samples_fd = np.arange(len(visual_focus_displacement))
            # Interpolate missing values
           # f_vis_fd = interp.interp1d(eye_samples_fd[valid_vis_fd], visual_focus_displacement[valid_vis_fd], kind=interp_kind, fill_value='extrapolate')

            # Assuming visual_focus_displacement and interp_kind are defined
            valid_vis_fd = ~np.isnan(visual_focus_displacement)
            eye_samples_fd = np.arange(len(visual_focus_displacement))
            
            # Interpolate missing values
            f_vis_fd = interp.interp1d(eye_samples_fd[valid_vis_fd], visual_focus_displacement[valid_vis_fd], kind=interp_kind, fill_value='extrapolate')
            vis_fd_filled = f_vis_fd(eye_samples_fd)
            
            # Calculate the standard deviation of the entire time series
            std_vis_fd = np.nanstd(visual_focus_displacement)
            
            # Cap the interpolated values to 2 times the standard deviation
            cap_value = 2 * std_vis_fd
            
            # Identify the interpolated indices (i.e., where the original data was NaN)
            interpolated_indices = np.isnan(visual_focus_displacement)
            
            # Create a copy of vis_fd_filled to apply the cap only on interpolated values
            vis_fd_filled_capped = vis_fd_filled.copy()
            vis_fd_filled_capped[interpolated_indices] = np.clip(vis_fd_filled[interpolated_indices], -cap_value, cap_value)
            
            # Example plot to visualize the capped values (optional)
            plt.figure(figsize=(12, 6))
            plt.plot(eye_samples_fd, visual_focus_displacement, label='Original Visual Focus Displacement', alpha=0.5)
            plt.plot(eye_samples_fd, vis_fd_filled, label='Interpolated Visual Focus Displacement', alpha=0.5)
            plt.plot(eye_samples_fd, vis_fd_filled_capped, label='Capped Visual Focus Displacement', alpha=0.8)
            plt.xlabel('Samples')
            plt.ylabel('Visual Focus Displacement')
            plt.title(f' {pat} Visual Focus Displacement with Interpolation and Capping of Interpolated Values')
            plt.legend()
            plt.grid(True)
            plt.close()

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(t_nwb, vis_fd_filled, label='Visual Focus Displacement (Interpolated)')
            plt.xlabel('Time')
            plt.ylabel('Displacement (mm)')
            plt.title('Visual Focus Displacement Over Time (Interpolated)')
            ax.legend()
            ax.set_title(f'Patient {pat} Vis Focus Disparity (interpolated)')
            fig.tight_layout()
            fig_filename = os.path.join(sub_et_prep_dir, f"{mov.replace('_ieeg.nwb', '')}_vf_disp_plot_interp.png")
            fig.savefig(fig_filename)
            plt.close(fig)  
            
            df = pd.DataFrame({
            'time': t_nwb,
            'gaze_dist_x_raw': x_diff,
            'gaze_dist_x_interp': x_diff_filled,
            'dva_gaze_disp_x': gaze_disparity_x,
            'dva_gaze_disp_x_interp': x_disp_filled,
            'vis_focus_disp': visual_focus_displacement,
            'vis_fd_interp': vis_fd_filled_capped,
            })

            # Save to csv
            et_prep_filename = '{:s}/{:s}'.format(sub_et_prep_dir, mov.replace('_ieeg.nwb', '_et_prep.csv'))

            csv_filename = et_prep_filename  # Replace with your desired output filename
            df.to_csv(csv_filename, index=False)  
            

io.close()     

