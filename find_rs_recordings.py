#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 13
Find which patients already have a resting state fixation recording preprocessed and which are needed

"""
import os
from itertools import compress
import numpy as np
import pandas as pd

#data_dir = '/Volumes/Expansion/Movie_data/movies_prep_standard'
data_dir = '/media/christine/Expansion/Movie_data/movies_prep_standard'

# List all patients in the data directory and sort them
patients = os.listdir(data_dir)
patients.sort()

pats_with_fix = []
pats_without_fix = []

for pat in patients:

    # Set patient ID and directories
    pat_dir = os.path.join(data_dir, pat)
    cx_pat_dir = os.path.join(data_dir, pat, 'Neural_prep')
    cx_files = os.listdir(cx_pat_dir)

    # Filter files to check if they contain the word 'fixation'
    cx_files_with_fixation = [f for f in cx_files if 'fixation' in f.lower()]

    if cx_files_with_fixation:
        pats_with_fix.append(pat)
    else:
        pats_without_fix.append(pat)
