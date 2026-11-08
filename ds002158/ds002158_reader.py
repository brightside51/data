# Library Imports
import os
import json
import re
import random
import argparse
import subprocess
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import mne

# Function Imports
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from natsort import natsorted
from scipy.fft import dctn, idctn
from torch.utils.data import Dataset

# ============================================================================================

# Simultaneous EEG-fMRI Dataset Reader Class (ds002158 | CN-EPFL Perceptual Confidence, 1 Session, 'main' Task Runs)
#   (Raw 5 kHz BrainVision EEG: 63 EEG Channels + ECG on Channel 32; TR 1.28 s, 108 x 108 x 64 Volumes, Multiband 4)
#   (The 'rest' Runs have no EEG Recording, so only 'main' Runs are used)
class DataReader(Dataset):

    # Runs Excluded by Default (none: the 'rejected/' Subfolders are never listed, and Runs without EEG are skipped)
    #   sub-14 main 005 (func/rejected): its Events File is a Copy of Run 006's, so its EEG Pairing cannot be checked
    #   sub-20 main 007 (EEG in eeg/rejected): EEG matches its Events, but the Authors left it out of their Analysis
    exclude_runs = ['sub-14_main_006']

    # EEG Files whose BIDS Label does not match their BOLD Run (BOLD Run -> EEG File Stem, relative to the 'eeg' Folder)
    #   (verified by matching the EEG Stimulus Markers to the Events Onsets, 48 / 48 Trials at 0 s Lag, and for sub-19
    #    by the Head Motion of the raw BOLD vs the Authors' Realignment Parameters, whose fmriprep Runs are 002-006)
    #   sub-12: 'run-00' holds s12_run5 (-> BOLD 005), 'run-005' holds s12_run6 (-> BOLD 006)
    #   sub-14: 'run-006' holds s14_run5 (matches no Events), 'rejected/run-005' holds s14_run6 (-> BOLD 006)
    #   sub-19: EEG 'run-001' is an aborted Scan (285 of 373 Volume Markers, no BOLD), so BOLD 00k <- EEG 00(k+1)
    eeg_files = {
        'sub-12_main_005': 'sub-12_ses-001_task-main_run-00',
        'sub-12_main_006': 'sub-12_ses-001_task-main_run-005',
        'sub-14_main_006': 'rejected/sub-14_ses-001_task-main_run-005',
        'sub-19_main_001': 'sub-19_ses-001_task-main_run-002',
        'sub-19_main_002': 'sub-19_ses-001_task-main_run-003',
        'sub-19_main_003': 'sub-19_ses-001_task-main_run-004',
        'sub-19_main_004': 'sub-19_ses-001_task-main_run-005',
        'sub-19_main_005': 'sub-19_ses-001_task-main_run-006',
    }

    # Events Files whose BIDS Label does not match their BOLD Run (sub-19 Events follow the EEG Labels, as in fmriprep)
    events_files = {
        'sub-19_main_001': 'sub-19_ses-001_task-main_run-002',
        'sub-19_main_002': 'sub-19_ses-001_task-main_run-003',
        'sub-19_main_003': 'sub-19_ses-001_task-main_run-004',
        'sub-19_main_004': 'sub-19_ses-001_task-main_run-005',
        'sub-19_main_005': 'sub-19_ses-001_task-main_run-006',
    }

    # Constructor / Initialisation
    def __init__(
        self,
        args: argparse.ArgumentParser,
        mode: str = 'train',
    ):

        # Argument Initialisation
        super().__init__()
        self.args = args; self.mode = mode
        self.prep_fp = Path(getattr(self.args, 'prep_fp', f"{self.args.data_fp}_prep"))
        self.cache = OrderedDict(); self.cache_size = getattr(self.args, 'cache_runs', 4)

        # Data Split Access / Creation
        split_fp = Path(f"{self.args.reader_fp}/{self.args.dataV}_{self.mode}.txt")
        if self.args.verbose: print(f"Split Filepath | {split_fp}")
        if split_fp.exists():
            if self.args.verbose: print(f"Reading Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.full_idx = split_fp.read_text().splitlines()
        else:
            if self.args.verbose: print(f"Creating Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.create_split()
            self.full_idx = list(getattr(self, f"{self.mode}_idx"))

        # --------------------------------------------------------------------------------------------

        # Sample Index (Run, Volume) & EEG Normalisation Statistics
        self.build_index()
        self.eeg_min, self.eeg_max = self.eeg_norm_stats()

    # ============================================================================================

    # DataLoader Length / No. Samples (Volumes) Computation
    def __len__(self): return len(self.sample_idx)

    # --------------------------------------------------------------------------------------------

    # Data Split Creation
    def create_split(self):

        # Patient ID List Fetching
        self.full_idx = []
        self.subj_list = natsorted([d for d in next(os.walk(self.args.data_fp))[1] if d.startswith('sub-')])
        # Acquisition Run Selection (as ds003768's state_list): only 'main' has EEG
        task_list = getattr(self.args, 'task_list', ['main'])
        exclude = self.exclude_runs + (getattr(self.args, 'exclude_runs', None) or [])
        if self.args.verbose: print(self.subj_list)
        for subj in self.subj_list:
            func_dir = Path(f"{self.args.data_fp}/{subj}/ses-001/func")

            # Task Run ID Fetching (only Runs with both BOLD & EEG; 'rejected/' Subfolders are not listed)
            for filename in os.listdir(func_dir):
                match = re.search(r"sub-(\d+)_ses-001_task-(\w+)_run-(\d+)_bold\.nii\.gz$", filename)
                if match:
                    subj_idx, task, run_idx = match.groups(); run_id = f"sub-{subj_idx}_{task}_{run_idx}"
                    if task not in task_list or run_id in exclude: continue
                    if Path(f"{self.eeg_fp(run_id)}.vhdr").exists(): self.full_idx.append(run_id)
                    elif self.args.verbose: print(f"Skipped Run | {run_id} | no EEG Recording")
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

    # Run ID Parsing Function ('sub-02_main_001' -> ['02', 'main', '001'])
    @staticmethod
    def parse_run(run_id: str):
        return re.search(r"sub-(\d+)_(\w+)_(\d+)$", run_id).groups()

    # BIDS Filepath Prefix Function (fMRI 'func' Folder)
    def run_fp(self, run_id: str):
        subj_idx, task, run_idx = self.parse_run(run_id)
        return f"{self.args.data_fp}/sub-{subj_idx}/ses-001/func/sub-{subj_idx}_ses-001_task-{task}_run-{run_idx}"

    # EEG Filepath Prefix Function (BIDS Label unless remapped in eeg_files)
    def eeg_fp(self, run_id: str):
        subj_idx, task, run_idx = self.parse_run(run_id)
        stem = self.eeg_files.get(run_id, f"sub-{subj_idx}_ses-001_task-{task}_run-{run_idx}")
        return f"{self.args.data_fp}/sub-{subj_idx}/ses-001/eeg/{stem}_eeg"

    # Events Filepath Function (BIDS Label unless remapped in events_files)
    def events_fp(self, run_id: str):
        subj_idx, task, run_idx = self.parse_run(run_id)
        stem = self.events_files.get(run_id, f"sub-{subj_idx}_ses-001_task-{task}_run-{run_idx}")
        return Path(f"{self.args.data_fp}/sub-{subj_idx}/ses-001/func/{stem}_events.tsv")

    # Volume Trigger Fetching Function (fMRI Volume 'R128' Marker Positions, 1-based in the .vmrk -> 0-based Samples)
    def get_triggers(self, run_id: str):
        vmrk = Path(f"{self.eeg_fp(run_id)}.vmrk").read_text(encoding = 'utf-8', errors = 'ignore')
        return np.array([int(p) - 1 for p in re.findall(r"Mk\d+=[^,]*,R128,(\d+)", vmrk)])

    # Sample Index Building Function (EEG Validity Check for Each fMRI Volume)
    def build_index(self):
        num_window = self.args.eeg.num_channel
        bold_shift = getattr(self.args.fmri, 'bold_shift', 6)
        self.sample_idx = []; self.run_meta = {}
        for run_id in self.full_idx:
            info = self.run_info(run_id)
            self.run_meta[run_id] = info
            for v in range(bold_shift + num_window, info['n_vol']):
                if info['eeg_valid'][v - num_window:v].all():
                    self.sample_idx.append((run_id, v))
        if self.args.verbose:
            print(f"{len(self.sample_idx)} Samples from {len(self.full_idx)} Runs | {self.mode} Set")

    # --------------------------------------------------------------------------------------------

    # Volume-Marker Pairing Function (the EEG Recording starts before the Scan, so the first Marker is the first fMRI
    # Volume; every used Run has exactly one Marker per Volume, but only the first min(n_marker, n_vol) are paired
    # in case of a missing / extra trailing Marker, and a Volume is valid if a full TR of EEG follows its Marker)
    @staticmethod
    def pair_volumes(triggers, n_vol: int, tr_samples: int, n_samp: int):
        n_pair = min(len(triggers), n_vol)
        eeg_valid = np.zeros(n_vol, dtype = bool)
        eeg_valid[:n_pair] = triggers[:n_pair] + tr_samples <= n_samp
        return eeg_valid, n_pair

    # Cheap Run Information Fetching Function (fMRI Volume Count, Repetition Time, EEG Sampling Frequency, EEG Validity)
    def run_info(self, run_id: str):
        fmri_fp = self.run_fp(run_id)
        n_vol = nib.load(f"{fmri_fp}_bold.nii.gz").shape[-1]
        tr = json.loads(Path(f"{fmri_fp}_bold.json").read_text())['RepetitionTime']

        # EEG Coverage per Volume (Header & Markers only, the .eeg Data is not loaded)
        eeg_fp = self.eeg_fp(run_id)
        vhdr = Path(f"{eeg_fp}.vhdr").read_text(encoding = 'utf-8', errors = 'ignore')
        num_channel = int(re.search(r"NumberOfChannels=(\d+)", vhdr).group(1))
        samp_freq = 1e6 / float(re.search(r"SamplingInterval=([\d.]+)", vhdr).group(1))
        n_bytes = {'INT_16': 2, 'INT_32': 4, 'IEEE_FLOAT_32': 4}[re.search(r"BinaryFormat=(\w+)", vhdr).group(1)]
        n_samp = os.path.getsize(f"{eeg_fp}.eeg") // (n_bytes * num_channel)
        triggers = self.get_triggers(run_id)
        eeg_valid, n_pair = self.pair_volumes(triggers, n_vol, int(round(tr * samp_freq)), n_samp)
        return {'n_vol': n_vol, 'tr': tr, 'samp_freq': samp_freq, 'eeg_valid': eeg_valid,
                'n_marker': len(triggers), 'n_pair': n_pair}

    # ============================================================================================

    # Single EEG Sample Fetching (Raw 5 kHz Recording: 63 EEG Channels + ECG on Channel 32)
    #   (the .vhdr 'DataFile' / 'MarkerFile' still name the original Recordings, e.g. 's2_run1_08062017.eeg',
    #    so the renamed BIDS Files are passed as MNE Header Overrides)
    def get_EEG(self, run_id: str):
        eeg_fp = self.eeg_fp(run_id)
        eeg_raw = mne.io.read_raw_brainvision(f"{eeg_fp}.vhdr", preload = True, verbose = 'error',
                                              overrides = {'data_fname': f"{eeg_fp}.eeg", 'marker_fname': f"{eeg_fp}.vmrk"})
        eeg_raw.set_channel_types({'ECG': 'ecg'}, verbose = 'error')
        return eeg_raw

    # --------------------------------------------------------------------------------------------

    # Single fMRI Batch Fetching
    def get_fMRI(self, run_id: str):
        fmri_fp = self.run_fp(run_id)

        # Nibabel Subject File Reading
        if self.args.fmri.data_format == 'nib':
            with open(f"{fmri_fp}_bold.json", "r") as f: fmri_meta = json.load(f)
            fmri_data = nib.load(f"{fmri_fp}_bold.nii.gz")
            fmri_img = fmri_data.get_fdata(dtype = np.float32)
            return fmri_meta, fmri_img

        else: raise(NotImplementedError)

    # ============================================================================================

    # Sliding Template Function (EEG Smoothing / Averaging)
    @staticmethod
    def sliding_template(epochs, window: int):          # [n_epoch, n_channel, n_sample]
        n = epochs.shape[0]; half = window // 2
        csum = np.concatenate([np.zeros_like(epochs[:1]), np.cumsum(epochs, axis = 0)], axis = 0)
        lo = np.clip(np.arange(n) - half, 0, None)
        hi = np.clip(np.arange(n) + half + 1, None, n)
        return (csum[hi] - csum[lo]) / (hi - lo)[:, None, None]

    # Gradient Removal Function (EEG Preprocessing)
    def remove_gradient(self, eeg_raw, triggers, tr_samples: int):
        data = eeg_raw._data
        trig = triggers[triggers + tr_samples <= data.shape[1]]
        idx = trig[:, None] + np.arange(tr_samples)[None, :]                       # [n_vol, L]
        epochs = data[:, idx].transpose(1, 0, 2).astype(np.float32)                 # [n_vol, ch, L]
        epochs -= epochs.mean(axis = 2, keepdims = True)
        template = self.sliding_template(epochs, getattr(self.args.eeg, 'grad_window', 21))
        for v, t in enumerate(trig): data[:, t:t + tr_samples] -= template[v]
        qc = {'grad_template_ptp_uV': float(np.median(np.ptp(template, axis = 2)) * 1e6)}
        return eeg_raw, len(trig), qc

    # --------------------------------------------------------------------------------------------

    # R-Peak Refractory Cleaning Function (EEG Preprocessing)
    @staticmethod
    def clean_peaks(peaks, amp, min_frac: float):
        thr = min_frac * np.median(np.diff(peaks)); kept = [peaks[0]]
        for p in peaks[1:]:
            if p - kept[-1] >= thr: kept.append(p)
            elif amp[p] > amp[kept[-1]]: kept[-1] = p
        return np.array(kept)

    # BCG Artifact Removal Function (EEG Preprocessing)
    #   (R-peak Channel set by eeg.bcg_channel: 'ECG' by default, 'TP10' as the Authors used in BrainVision Analyzer)
    def remove_bcg(self, eeg_raw):
        samp_freq = eeg_raw.info['sfreq']; eeg_picks = mne.pick_types(eeg_raw.info, eeg = True)
        bcg_channel = getattr(self.args.eeg, 'bcg_channel', 'ECG')

        # R-Peak Detection on the (gradient-corrected) Reference Channel
        ecg_events, _, _ = mne.preprocessing.find_ecg_events(eeg_raw, ch_name = bcg_channel, verbose = 'error')
        peaks_raw = ecg_events[:, 0] - eeg_raw.first_samp
        amp = np.abs(mne.filter.filter_data(eeg_raw.get_data(picks = bcg_channel)[0], samp_freq, 5., 35., verbose = 'error'))
        peaks = self.clean_peaks(peaks_raw, amp, getattr(self.args.eeg, 'bcg_min_rr', 0.6))
        rr = np.diff(peaks); med_rr = int(np.median(rr))
        pre = int(self.args.eeg.bcg_pre * med_rr); post = med_rr - pre
        peaks = peaks[(peaks - pre >= 0) & (peaks + post <= eeg_raw.n_times)]

        # Template Subtraction (only up to the next beat's segment start, so overlaps are not doubled)
        data = eeg_raw.get_data(picks = eeg_picks)
        epochs = data[:, peaks[:, None] + np.arange(-pre, post)[None, :]].transpose(1, 0, 2)
        template = self.sliding_template(epochs - epochs.mean(axis = 2, keepdims = True), self.args.eeg.bcg_window)
        for b, p in enumerate(peaks):
            stop = post if b + 1 == len(peaks) else min(post, peaks[b + 1] - pre - p)
            data[:, p - pre:p + stop] -= template[b][:, :pre + stop]
        eeg_raw._data[eeg_picks] = data
        qc = {'bcg_channel': bcg_channel, 'heart_rate_bpm': float(60 * samp_freq / med_rr), 'n_beats': int(len(peaks)),
              'n_beats_removed': int(len(peaks_raw) - len(peaks)), 'rr_cv': float(rr.std() / rr.mean()),
              'rr_frac_long': float((rr > 1.5 * med_rr).mean()),
              'bcg_template_ptp_uV': float(np.median(np.ptp(template, axis = 2)) * 1e6)}
        return eeg_raw, qc

    # Spectral Feature Extraction Function (EEG Preprocessing)
    #   (TR 1.28 s -> 320-Sample Windows at 250 Hz, 0.78 Hz Bins; num_sample 159 keeps the 0.8-125 Hz Band of the 2 s Datasets)
    def eeg_features(self, eeg_raw, starts, valid, tr: float):
        samp_freq = eeg_raw.info['sfreq']; win = int(round(tr * samp_freq)); num_freq = self.args.eeg.num_sample
        assert num_freq < win // 2, f"ERROR | {num_freq} Frequency Bins exceed Nyquist for a {win}-Sample Window"
        eeg = eeg_raw.get_data(picks = mne.pick_types(eeg_raw.info, eeg = True)).astype(np.float32)
        starts = np.minimum(starts, eeg.shape[1] - win)                             # resampling round-off
        feats = np.zeros((len(valid), eeg.shape[0], num_freq), dtype = np.float32)
        for v in np.where(valid)[0]:
            feats[v] = np.abs(np.fft.fft(eeg[:, starts[v]:starts[v] + win], axis = 1))[:, 1:num_freq + 1]
        qc = {'freq_res_hz': samp_freq / win, 'freq_max_hz': samp_freq / win * num_freq}
        return feats, qc                                                            # [n_vol, C, F]

    # --------------------------------------------------------------------------------------------

    # Full EEG Run Preprocessing Function (Gradient AAS -> Crop -> Resample -> High-pass -> BCG AAS -> Features)
    def prep_EEG(self, run_id: str, n_vol: int, tr: float):

        # Raw Reading & Volume Markers
        eeg_raw = self.get_EEG(run_id)
        samp_freq = eeg_raw.info['sfreq']; tr_samples = int(round(tr * samp_freq))
        triggers = self.get_triggers(run_id)
        valid, n_pair = self.pair_volumes(triggers, n_vol, tr_samples, eeg_raw.n_times)

        # Gradient AAS on all Channels (incl. ECG, needed clean for R-peak detection)
        eeg_raw, n_grad, grad_qc = self.remove_gradient(eeg_raw, triggers, tr_samples)

        # Crop to Scan Period, Resample (built-in anti-aliasing) & High-pass
        t0 = triggers[0]; t_end = triggers[n_grad - 1] + tr_samples
        eeg_raw.crop(tmin = t0 / samp_freq, tmax = (t_end - 1) / samp_freq)
        eeg_raw.resample(self.args.eeg.samp_freq, verbose = 'error')
        eeg_raw.filter(l_freq = self.args.eeg.hpass, h_freq = None, picks = 'all', verbose = 'error')

        # BCG AAS on EEG Channels & Spectral Features (Volumes without a Marker get a placeholder Start, never used)
        eeg_raw, bcg_qc = self.remove_bcg(eeg_raw)
        starts = np.round((triggers[:n_pair] - t0) / samp_freq * eeg_raw.info['sfreq']).astype(int)
        starts = np.concatenate([starts, np.zeros(n_vol - n_pair, dtype = int)])
        feats, feat_qc = self.eeg_features(eeg_raw, starts, valid, tr)
        qc = {'eeg_file': Path(self.eeg_fp(run_id)).name,
              'eeg_channels': [eeg_raw.ch_names[i] for i in mne.pick_types(eeg_raw.info, eeg = True)],
              'n_marker': int(len(triggers)), 'n_pair': int(n_pair), 'n_valid_volumes': int(valid.sum()),
              **grad_qc, **bcg_qc, **feat_qc}
        return feats, valid, qc

    # ============================================================================================

    # fMRI Preprocessing Function (In-plane DCT Low-pass Resize, Ringing below 0 Clipped)
    #   (done in Chunks of Volumes, as a full 108 x 108 x 64 x 373 Run is ~1.1 GB in float32)
    #   (Run divided by its Maximum first, as in ds007216, so the float16 Cache can never overflow;
    #    Volumes are min-max normalised when fetched, so only the Scale Factor (saved in the Meta File) is lost)
    def prep_fMRI(self, fmri_img, chunk: int = 64):     # [X, Y, Z, T] -> [T, Z, img_size, img_size]
        X, Y, Z, T = fmri_img.shape; img_size = self.args.fmri.img_size
        scale = float(fmri_img.max()) or 1.0; fmri_img = fmri_img / scale
        fmri = np.zeros((T, Z, img_size, img_size), dtype = np.float16)
        for t in range(0, T, chunk):
            coef = dctn(fmri_img[..., t:t + chunk], axes = (0, 1), norm = 'ortho')[:img_size, :img_size]
            vol = idctn(coef, axes = (0, 1), norm = 'ortho') * np.sqrt(img_size * img_size / (X * Y))
            fmri[t:t + chunk] = np.clip(vol, 0, None).transpose(3, 2, 0, 1)
        assert np.isfinite(fmri).all(), "ERROR | Non-finite fMRI Values after Resizing"
        return fmri, scale

    # ============================================================================================

    # Task Condition Fetching Function (Per-Volume Block Label from the BIDS Events File, by Volume Onset Time)
    #   'act' (Committed Decisions) / 'obs' (Observed Decisions): Trials come in Blocks, each Event is a single Trial
    #   (Duration 0, 'act' / 'act-bad' / 'obs' / 'obs-bad'), so a Volume takes the Condition of the latest Trial
    #   started at or before its Onset; 'n/a' before the first Trial (Onsets in s from the first Volume Marker)
    def get_conditions(self, run_id: str, n_vol: int, tr: float):
        tsv = self.events_fp(run_id)
        if not tsv.exists(): return ['n/a'] * n_vol
        lines = tsv.read_text().splitlines(); head = lines[0].strip().split('\t')
        c_on, c_type = head.index('onset'), head.index('trial_type')
        trials = []
        for line in lines[1:]:
            cols = line.strip().split('\t')
            if len(cols) == len(head): trials.append((float(cols[c_on]), cols[c_type].split('-')[0]))
        trials.sort(); onsets = np.array([on for on, _ in trials])
        last = np.searchsorted(onsets, np.arange(n_vol) * tr, side = 'right') - 1
        return [trials[i][1] if i >= 0 else 'n/a' for i in last]

    # ============================================================================================

    # Preprocessed Run Filepath Function
    def prep_files(self, run_id: str):
        subj_idx, _, _ = self.parse_run(run_id); prep_dir = self.prep_fp / f"sub-{subj_idx}"
        return {k: prep_dir / f"{run_id}_{k}.{'json' if k == 'meta' else 'npy'}" for k in ('eeg', 'fmri', 'meta')}

    # Full Run Preprocessing Function (+ Saving if save_prep)
    def prep_run(self, run_id: str, save: bool = None):
        save = self.args.save_prep if save is None else save
        if self.args.verbose: print(f"Preprocessing Run | {run_id}")
        fmri_meta, fmri_img = self.get_fMRI(run_id)
        n_vol, tr = fmri_img.shape[-1], fmri_meta['RepetitionTime']
        fmri, fmri_scale = self.prep_fMRI(fmri_img); del fmri_img
        eeg, valid, qc = self.prep_EEG(run_id, n_vol, tr)
        meta = {'run_id': run_id, 'n_vol': n_vol, 'tr': tr, 'eeg_shape': list(eeg.shape), 'fmri_shape': list(fmri.shape),
                'fmri_scale': fmri_scale, 'eeg_valid': valid.tolist(), 'condition': self.get_conditions(run_id, n_vol, tr),
                'qc': qc}

        # Preprocessed Run Saving (with the parameters, code version & library versions used)
        if save:
            files = self.prep_files(run_id); files['eeg'].parent.mkdir(parents = True, exist_ok = True)
            np.save(files['eeg'], eeg); np.save(files['fmri'], fmri)
            with open(files['meta'], 'w') as f:
                json.dump({**meta, 'args': {'eeg': vars(self.args.eeg), 'fmri': vars(self.args.fmri)},
                           'git_commit': self.git_commit(), 'created': datetime.now().isoformat(timespec = 'seconds'),
                           'versions': {'mne': mne.__version__, 'numpy': np.__version__, 'nibabel': nib.__version__}},
                          f, indent = 1, default = str)
        return {'eeg': eeg, 'fmri': fmri, 'meta': meta}

    # Run Loading Function (RAM Cache -> Saved Preprocessed Files (Memory-mapped) -> Preprocess Now)
    def load_run(self, run_id: str):
        if run_id in self.cache: self.cache.move_to_end(run_id); return self.cache[run_id]
        files = self.prep_files(run_id)
        if files['meta'].exists():
            run = {'eeg': np.load(files['eeg'], mmap_mode = 'r'), 'fmri': np.load(files['fmri'], mmap_mode = 'r'),
                   'meta': json.loads(files['meta'].read_text())}
        else: run = self.prep_run(run_id)
        self.cache[run_id] = run
        if len(self.cache) > self.cache_size: self.cache.popitem(last = False)
        return run

    # Split Preprocessing Function (every Run of this Split, or a single one e.g. idx = $SLURM_ARRAY_TASK_ID)
    def prepare(self, idx: int = None, overwrite: bool = False):
        for run_id in (self.full_idx if idx is None else [self.full_idx[idx]]):
            if overwrite or not self.prep_files(run_id)['meta'].exists(): self.prep_run(run_id, save = True)

    # Git Commit Fetching Function (Code Version saved with the Preprocessed Data)
    @staticmethod
    def git_commit():
        try: return subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'],
                                            stderr = subprocess.DEVNULL, text = True).strip()
        except Exception: return None

    # Run-Mean Volume Function (mean of the per-volume min-max normalised target volumes of a run, i.e.
    # its static anatomy; used by the fluctuation metrics, never as model input). Kept in its own dict
    # (~1 MB per run), not the LRU run cache, so shuffled batches never recompute it
    def run_mean(self, run_id: str):
        if not hasattr(self, 'run_means'):
            self.run_means = {}; self.run_vols = {}
            for r, v in self.sample_idx: self.run_vols.setdefault(r, []).append(v)
        if run_id not in self.run_means:
            run = self.load_run(run_id); vols = sorted(self.run_vols[run_id]); total = 0
            for i in range(0, len(vols), 64):
                f = np.asarray(run['fmri'][vols[i:i + 64]], dtype = np.float32)
                mn = f.min(axis = (1, 2, 3), keepdims = True); mx = f.max(axis = (1, 2, 3), keepdims = True)
                total = total + ((f - mn) / (mx - mn + 1e-10)).sum(0)
            self.run_means[run_id] = (total / max(len(vols), 1)).astype(np.float32)
        return self.run_means[run_id]

    # ============================================================================================

    # EEG Z-Score Function (across Electrodes per Window & Frequency Bin, as E2fNet's scipy zscore axis 0)
    def eeg_zscore(self, eeg):
        if not getattr(self.args.eeg, 'zscore', True): return eeg
        return (eeg - eeg.mean(axis = -2, keepdims = True)) / (eeg.std(axis = -2, keepdims = True) + 1e-10)

    # EEG Normalisation Statistics Function (Global Min-Max on the Train Split only, saved for Val / Test)
    def eeg_norm_stats(self):
        stats_fp = Path(f"{self.args.reader_fp}/{self.args.dataV}_eeg_norm.json")
        if stats_fp.exists(): stats = json.loads(stats_fp.read_text()); return stats['min'], stats['max']
        assert self.mode == 'train', f"ERROR | {stats_fp} not found: build the train set first"
        if self.args.verbose: print(f"Computing EEG Normalisation Statistics | {len(self.full_idx)} Train Runs")
        eeg_min, eeg_max = np.inf, -np.inf
        for run_id in self.full_idx:
            run = self.load_run(run_id); valid = np.array(run['meta']['eeg_valid'])
            eeg = self.eeg_zscore(np.asarray(run['eeg'])[valid])
            eeg_min, eeg_max = min(eeg_min, float(eeg.min())), max(eeg_max, float(eeg.max()))
        with open(stats_fp, 'w') as f: json.dump({'min': eeg_min, 'max': eeg_max, 'runs': list(self.full_idx)}, f, indent = 1)
        return eeg_min, eeg_max

    # ============================================================================================

    # EEG Plotting Function (Mean TR-Window Spectrum of a Preprocessed Run)
    def plot_EEG(self, run_id: str, channel: int = 0):
        run = self.load_run(run_id); valid = np.array(run['meta']['eeg_valid'])
        freq = (np.arange(self.args.eeg.num_sample) + 1) * run['meta']['qc']['freq_res_hz']
        plt.figure(figsize = (12, 6))
        plt.semilogy(freq, np.asarray(run['eeg'])[valid, channel].mean(0))
        plt.title(f"{run_id} | {run['meta']['qc']['eeg_channels'][channel]} | Mean TR-Window Spectrum")
        plt.xlabel('Frequency (Hz)'); plt.ylabel('|FFT| (V)'); plt.grid(True); plt.show()

    # ============================================================================================

    # Single Sample Fetching (EEG [num_channel Windows, C, F] -> fMRI Volume [Z, img_size, img_size])
    def __getitem__(self, idx: int):
        run_id, v = self.sample_idx[idx]; num_window = self.args.eeg.num_channel
        run = self.load_run(run_id)

        # Sample Data Fetching | EEG (Z-Score across Electrodes, then Global Min-Max with Train Statistics)
        eeg = self.eeg_zscore(np.asarray(run['eeg'][v - num_window:v], dtype = np.float32))
        eeg = (eeg - self.eeg_min) / (self.eeg_max - self.eeg_min + 1e-10)

        # Sample Data Fetching | fMRI (Min-Max per Volume)
        fmri = np.asarray(run['fmri'][v], dtype = np.float32)
        fmri = (fmri - fmri.min()) / (fmri.max() - fmri.min() + 1e-10)

        # Sample Data Fetching | Run-Mean Volume & Repetition Time (for the fluctuation & run-level metrics)
        fmri_mean = self.run_mean(run_id)

        # 'sleep_stage' kept ('n/a') as the E2fNet Trainer groups its Test Breakdown by it; 'condition' is this Dataset's Label
        return {'eeg': torch.from_numpy(eeg), 'fmri': torch.from_numpy(fmri), 'fmri_mean': torch.from_numpy(fmri_mean),
                'tr': float(run['meta']['tr']), 'run_id': run_id, 'volume': v,
                'condition': run['meta']['condition'][v], 'sleep_stage': 'n/a'}

    # ============================================================================================
