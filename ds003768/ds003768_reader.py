# Library Imports
import os
import json
import re
import random
import argparse
import natsort
import numpy as np
import pydicom
import torch
import torchvision
import matplotlib.pyplot as plt
import nibabel as nib

# Function Imports
from pathlib import Path
from natsort import natsorted
from ipywidgets import interactive, IntSlider
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ============================================================================================

# Simultaneous EEG-fMRI Dataset Reader Class
class DataReader(Dataset):

    # Constructor / Initialisation
    def __init__(
        self,
        args: argparse.ArgumentParser,
        mode: str = 'train',
    ):  
        
        # Argument Initialisation
        super().__init__()
        self.args = args; self.mode = mode

        # Data Split Access / Creation
        split_fp = Path(f"{self.args.reader_fp}/{self.args.dataV}_{self.mode}.txt")
        if self.args.verbose: print(f"Split Filepath | {split_fp}")
        if split_fp.exists():
            if self.args.verbose: print(f"Reading Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.full_idx = split_fp.read_text().splitlines()
        else:
            if self.args.verbose: print(f"Creating Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.create_split()
        
        # --------------------------------------------------------------------------------------------
        
        # Dataset Transformations Initialisation
        self.fmri_transform =    transforms.Compose([
                            transforms.Resize(( self.args.fmri.img_size,
                                                self.args.fmri.img_size)),
                            transforms.ToTensor()])
        self.fmri_h_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1)])
        self.fmri_v_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1)])

    # ============================================================================================

    # DataLoader Length / No. Subjects Computation
    def __len__(self): return len(self.full_idx)

    # --------------------------------------------------------------------------------------------

    # Data Split Creation
    def create_split(self):

        # Patient ID List Fetching
        self.full_idx = []
        self.subj_list = natsorted(next(os.walk(self.args.data_fp))[1])
        self.subj_list.remove('sourcedata')
        if self.args.verbose: print(self.subj_list)
        for subj in self.subj_list:
            subj_dir = Path(f"{self.args.data_fp}/{subj}/func")

            # Task Run ID Fetching
            for filename in os.listdir(subj_dir):
                task_dir, task_ext = os.path.splitext(filename)
                if task_ext == '.gz':
                    match = re.search(r"sub-(\d+)_task-(rest|sleep)_run-(\d+)", task_dir)
                    if match:
                        subj_idx, task, run_idx = match.groups()
                        #if self.args.verbose: print(f"Found Run | Subject: {subj_idx} | Task: {task} | Run: {run_idx}")
                        if task in self.args.state_list:
                            self.full_idx.append(f"sub-{subj_idx}_{task}_{run_idx}")
        print(f"{len(self.full_idx)} Runs for {len(self.subj_list)} Subjects Found")
        self.full_idx = natsorted(self.full_idx)
        if self.args.verbose: print(self.full_idx)

        # --------------------------------------------------------------------------------------------

        # Data Splits
        rest_idx = self.full_idx
        self.train_split = len(self.full_idx) - (self.args.val_split + self.args.test_split)
        assert 0 < self.train_split + self.args.val_split + self.args.test_split <= len(self.full_idx),\
            f"ERROR | Dataset Split Proportions are Invalid ({self.train_split} Train + {self.args.val_split} Val + {self.args.test_split} Test)"

        # Validation & Test Splits
        if self.args.val_split != 0:
            self.val_idx = np.sort(np.array(random.sample(rest_idx, self.args.val_split), dtype = 'str'))
            rest_idx = [subj for subj in rest_idx if subj not in self.val_idx]
        if self.args.test_split != 0:
            self.test_idx = np.sort(np.array(random.sample(rest_idx, self.args.test_split), dtype = 'str'))
            rest_idx = [subj for subj in rest_idx if subj not in self.test_idx]
        
        # Train & Remainder Splits
        self.train_idx = np.sort(np.array(random.sample(rest_idx, self.train_split), dtype = 'str'))
        rest_idx = [subj for subj in rest_idx if subj not in self.train_idx]
        self.rest_idx = np.sort(np.array(rest_idx, dtype = 'str'))
        assert len(self.train_idx) == self.train_split and len(self.rest_idx) + self.train_split + self.args.val_split + self.args.test_split == len(self.full_idx),\
            f"ERROR | Dataset Split Proportions are Invalid ({self.train_split} Train + {self.args.val_split} Val + {self.args.test_split} Test + {len(self.rest_idx)} Rest)"

        # Data Split Saving
        if len(self.train_idx) != 0: np.savetxt(f"{self.args.reader_fp}/{self.args.dataV}_train.txt", self.train_idx, fmt='%s')
        if len(self.rest_idx) != 0: np.savetxt(f"{self.args.reader_fp}/{self.args.dataV}_rest.txt", self.rest_idx, fmt='%s')
        if self.args.val_split != 0 and len(self.val_idx) != 0:
            np.savetxt(f"{self.args.reader_fp}/{self.args.dataV}_val.txt", self.val_idx, fmt='%s')
        if self.args.test_split != 0 and len(self.test_idx) != 0:
            np.savetxt(f"{self.args.reader_fp}/{self.args.dataV}_test.txt", self.test_idx, fmt='%s')

    # ============================================================================================

    # Single EEG Batch Fetching
    def get_EEG(self, subj_idx: int, task: str, run_idx: int):

        # EEG Information Fetching
        eeg_fp = Path(f"{self.args.data_fp}/sub-{subj_idx}/eeg/sub-{subj_idx}_task-{task}_run-{run_idx}_eeg")
        #eeg_bin = mne.io.read_raw_brainvision(f"{eeg_fp}.vhdr", preload = True)
        with open(f"{eeg_fp}.vhdr", 'r') as f: eeg_bin = f.read()
        with open(f"{eeg_fp}.vmrk", "r") as f: eeg_event = f.readlines()
        eeg_event = [line for line in eeg_event if line.startswith('Mk')]
        #print(f"Total number of raw marker entries: {len(eeg_event)}")

        #eeg_event = mne.events_from_annotations(mne.read_annotations(f"{eeg_fp}.vmrk"))
        eeg_data = np.fromfile(f"{eeg_fp}.eeg", dtype = np.int32)

        # EEG Data Processing
        #if self.args.eeg.num_channel == None:
        num_channel = int(re.search(r'NumberOfChannels=(.*)', eeg_bin).group(1))
        samp_interval = float(re.search(r'SamplingInterval=(.*)', eeg_bin).group(1))
        samp_freq = int(1e6 / samp_interval)
        num_sample = len(eeg_data) // num_channel
        print(f"EEG Data | {num_sample} Samples | {num_channel} Channels | {samp_freq} Hz Sampling Frequency")
        eeg_signal = eeg_data.reshape((num_sample, num_channel))

        return eeg_bin, eeg_event, eeg_data, eeg_signal

    # --------------------------------------------------------------------------------------------

    # Single fMRI Batch Fetching
    def get_fMRI(self, subj_idx: int, task: str, run_idx: int):
        fmri_fp = Path(f"{self.args.data_fp}/sub-{subj_idx}/func/sub-{subj_idx}_task-{task}_run-{run_idx}_bold")

        # Nibabel Subject File Reading
        if self.args.fmri.data_format == 'nib':
            with open(f"{fmri_fp}.json", "r") as f: fmri_meta = json.load(f)
            fmri_data = nib.load(f"{fmri_fp}.nii.gz")
            fmri_img = fmri_data.get_fdata()
            return fmri_meta, fmri_data, fmri_img
            
        else: raise(NotImplementedError)
            
    # ============================================================================================

    def plot_EEG(self, eeg_bin, eeg_signal):

        samp_interval = float(re.search(r'SamplingInterval=(.*)', eeg_bin).group(1))
        samp_freq = int(1e6 / samp_interval)
        samp_plot = [int(self.args.eeg.plot_interval[0] * samp_freq), int(self.args.eeg.plot_interval[1] * samp_freq)]
        time_x = np.arange(samp_plot[0], samp_plot[1]) / samp_freq

        # Channel Plotting with Offsets for Visibility
        plt.figure(figsize=(12, 6)); offset = 0 
        for channel in range(eeg_signal.shape[1]):
            plt.plot(time_x, eeg_signal[samp_plot[0]:samp_plot[1], channel] + offset, label=f'Channel {channel+1}')
            offset += 100 
        plt.title(f'EEG Data ({self.args.eeg.plot_interval} seconds)')
        plt.xlabel('Time (s)'); plt.ylabel('Amplitude (Offset for Visibility)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True); plt.show()

    # --------------------------------------------------------------------------------------------

    #def plot_fMRI

    # --------------------------------------------------------------------------------------------

    # ============================================================================================

   # Single Batch Fetching
    def __getitem__(self, idx: int = 0 or str, save: bool = False):

        # Sample Data Fetching | Index
        if type(idx) == int: idx = self.full_idx[idx]
        match = re.search(r"sub-(\d+)_(rest|sleep)_(\d+)", idx)
        subj_idx, task, run_idx = match.groups()
        if self.args.verbose: print(f"Fetching Data  | Patient {subj_idx} | Task {task} | Run {run_idx}")

        # Sample Data Fetching | fMRI & EEG
        fmri_meta, fmri_data, fmri_img = self.get_fMRI(subj_idx, task, run_idx)        #fmri_data.affine
        eeg_bin, eeg_event, eeg_data, eeg_signal = self.get_EEG(subj_idx, task, run_idx)

        # Sample Data Fetching | Anatomical
        with open(f"{self.args.data_fp}/sub-{subj_idx}/anat/sub-{subj_idx}_T1w.json", "r") as f: anat_meta = json.load(f)
        anat_data = nib.load(f"{self.args.data_fp}/sub-{subj_idx}/anat/sub-{subj_idx}_T1w.nii.gz")
        anat_img = anat_data.get_fdata()

        #
        
        return {'fmri_meta': fmri_meta, 'fmri_data': fmri_data, 'fmri_img': fmri_img,
                'eeg_bin': eeg_bin, 'eeg_event': eeg_event, 'eeg_data': eeg_data, 'eeg_signal': eeg_signal,
                'anat_meta': anat_meta, 'anat_data': anat_data, 'anat_img': anat_img}
        
    # ============================================================================================