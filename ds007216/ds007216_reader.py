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

# Simultaneous EEG-fMRI Dataset Reader Class (ds007216 | GradCPT & Resting-State Experience Sampling, 2 Sessions)
class DataReader(Dataset):

    # Runs Excluded by Default
    #   sub-003 ses-001 GradCPT: 218 EEG Markers vs 250 fMRI Volumes (32 missing, not covered by the README)
    #   sub-018 ses-002 ExperienceSampling 001: CWreg .fdt truncated at the Source (22344946 of 24541920 Bytes, also on OpenNeuro)
    exclude_runs = ['sub-003_ses-001_GradCPT_001', 'sub-018_ses-002_ExperienceSampling_001']

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
        # Acquisition Run Selection (as ds003768's state_list): Tasks ('ExperienceSampling' / 'GradCPT') & Sessions (1 / 2)
        state_list = getattr(self.args, 'state_list', ['ExperienceSampling', 'GradCPT'])
        ses_list = [int(s) for s in getattr(self.args, 'ses_list', [1, 2])]
        exclude = getattr(self.args, 'exclude_runs', self.exclude_runs)
        if self.args.verbose: print(self.subj_list)
        for subj in self.subj_list:
            for ses in natsorted(d for d in os.listdir(f"{self.args.data_fp}/{subj}") if d.startswith('ses-')):
                if int(ses[4:]) not in ses_list: continue
                func_dir = Path(f"{self.args.data_fp}/{subj}/{ses}/func")

                # Task Run ID Fetching (macOS '._' Resource-Fork Files skipped)
                for filename in os.listdir(func_dir):
                    if filename.startswith('._') or not filename.endswith('_bold.nii.gz'): continue
                    match = re.search(r"sub-(\d+)_ses-(\d+)_task-(\w+?)_run-(\d+)_bold", filename)
                    if match:
                        subj_idx, ses_idx, task, run_idx = match.groups()
                        run_id = f"sub-{subj_idx}_ses-{ses_idx}_{task}_{run_idx}"
                        if task in state_list and run_id not in exclude: self.full_idx.append(run_id)
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

    # Run ID Parsing Function ('sub-007_ses-001_ExperienceSampling_001' -> ['007', '001', 'ExperienceSampling', '001'])
    @staticmethod
    def parse_run(run_id: str):
        return re.search(r"sub-(\d+)_ses-(\d+)_(\w+?)_(\d+)$", run_id).groups()

    # BIDS Filepath Prefix Function (Modality Folder: 'func' / 'eeg', Raw or Derivatives Tree)
    def run_fp(self, run_id: str, modality: str, derivatives: bool = False):
        subj_idx, ses_idx, task, run_idx = self.parse_run(run_id)
        root = f"{self.args.data_fp}/derivatives" if derivatives else self.args.data_fp
        return f"{root}/sub-{subj_idx}/ses-{ses_idx}/{modality}/sub-{subj_idx}_ses-{ses_idx}_task-{task}_run-{run_idx}"

    # Preprocessed EEG Filepath Function (Gradient + CWL-Regression corrected EEGLAB Set, 250 Hz)
    #   (sub-023 ses-002 is named '..._eeg_events_events_Bergen_CWreg.set', hence the glob)
    def eeg_file(self, run_id: str):
        prefix = Path(self.run_fp(run_id, 'eeg', derivatives = True))
        files = [f for f in prefix.parent.glob(f"{prefix.name}_eeg*_Bergen_CWreg.set") if not f.name.startswith('._')]
        assert len(files) == 1, f"ERROR | {run_id}: {len(files)} CWreg EEG Files found"
        return files[0]

    # Volume Trigger Fetching Function (fMRI Volume 'T  1' Marker Positions, in Samples, from the EEG Annotations)
    @staticmethod
    def get_triggers(eeg_raw):
        ann = eeg_raw.annotations; sel = np.array([d == 'T  1' for d in ann.description], dtype = bool)
        return np.round(ann.onset[sel] * eeg_raw.info['sfreq']).astype(int) - eeg_raw.first_samp

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

    # Volume-Marker Pairing Function (Markers are aligned with the first fMRI Volume, as stated in the README;
    # only the trailing Markers may be extra (forced Scan Stop) or missing, so the first min(n_marker, n_vol) are paired)
    @staticmethod
    def pair_volumes(triggers, n_vol: int, tr_samples: int, n_samp: int):
        n_pair = min(len(triggers), n_vol)
        eeg_valid = np.zeros(n_vol, dtype = bool)
        eeg_valid[:n_pair] = triggers[:n_pair] + tr_samples <= n_samp
        return eeg_valid, n_pair

    # Cheap Run Information Fetching Function (fMRI Volume Count, Repetition Time, EEG Sampling Frequency, EEG Validity)
    def run_info(self, run_id: str):
        fmri_fp = self.run_fp(run_id, 'func')
        n_vol = nib.load(f"{fmri_fp}_bold.nii.gz").shape[-1]
        tr = json.loads(Path(f"{fmri_fp}_bold.json").read_text())['RepetitionTime']

        # EEG Coverage per Volume (Header & Annotations only, the .fdt Data is not loaded)
        eeg_raw = mne.io.read_raw_eeglab(self.eeg_file(run_id), preload = False, verbose = 'error')
        samp_freq = eeg_raw.info['sfreq']; triggers = self.get_triggers(eeg_raw)
        eeg_valid, n_pair = self.pair_volumes(triggers, n_vol, int(round(tr * samp_freq)), eeg_raw.n_times)
        return {'n_vol': n_vol, 'tr': tr, 'samp_freq': samp_freq, 'eeg_valid': eeg_valid,
                'n_marker': len(triggers), 'n_pair': n_pair}

    # ============================================================================================

    # Single EEG Sample Fetching (Authors' Gradient (Bergen) + CWL-Regression Corrected EEG)
    def get_EEG(self, run_id: str):
        eeg_raw = mne.io.read_raw_eeglab(self.eeg_file(run_id), preload = True, verbose = 'error')
        eeg_raw.set_channel_types({'ECG': 'ecg', **{f"CWL{i}": 'misc' for i in range(1, 5)}}, verbose = 'error')
        return eeg_raw

    # --------------------------------------------------------------------------------------------

    # Single fMRI Batch Fetching
    def get_fMRI(self, run_id: str):
        fmri_fp = self.run_fp(run_id, 'func')

        # Nibabel Subject File Reading
        if self.args.fmri.data_format == 'nib':
            with open(f"{fmri_fp}_bold.json", "r") as f: fmri_meta = json.load(f)
            fmri_data = nib.load(f"{fmri_fp}_bold.nii.gz")
            fmri_img = fmri_data.get_fdata(dtype = np.float32)
            return fmri_meta, fmri_img

        else: raise(NotImplementedError)

    # ============================================================================================

    # Spectral Feature Extraction Function (EEG Preprocessing)
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

    # Full EEG Run Preprocessing Function (CWreg EEG -> Crop -> Resample -> High-pass -> Features)
    #   (Gradient & BCG / Helium-Pump / Motion Artifacts already removed by the Authors: Bergen AAS + CWL Regression)
    def prep_EEG(self, run_id: str, n_vol: int, tr: float):

        # Corrected EEG Reading & Volume Markers
        eeg_raw = self.get_EEG(run_id)
        samp_freq = eeg_raw.info['sfreq']; tr_samples = int(round(tr * samp_freq))
        triggers = self.get_triggers(eeg_raw)
        valid, n_pair = self.pair_volumes(triggers, n_vol, tr_samples, eeg_raw.n_times)
        last = np.where(valid)[0][-1]

        # Crop to Scan Period, Resample (built-in anti-aliasing, no-op at 250 Hz) & High-pass
        t0 = triggers[0]; t_end = triggers[last] + tr_samples
        eeg_raw.crop(tmin = t0 / samp_freq, tmax = (t_end - 1) / samp_freq)
        eeg_raw.resample(self.args.eeg.samp_freq, verbose = 'error')
        eeg_raw.filter(l_freq = self.args.eeg.hpass, h_freq = None, picks = 'all', verbose = 'error')

        # Spectral Features
        starts = np.round((triggers[:n_pair] - t0) / samp_freq * eeg_raw.info['sfreq']).astype(int)
        starts = np.concatenate([starts, np.zeros(n_vol - n_pair, dtype = int)])
        feats, feat_qc = self.eeg_features(eeg_raw, starts, valid, tr)
        qc = {'eeg_file': self.eeg_file(run_id).name, 'eeg_channels': [eeg_raw.ch_names[i] for i in mne.pick_types(eeg_raw.info, eeg = True)],
              'n_marker': int(len(triggers)), 'n_pair': int(n_pair), 'n_valid_volumes': int(valid.sum()), **feat_qc}
        return feats, valid, qc

    # ============================================================================================

    # fMRI Preprocessing Function (In-plane DCT Low-pass Resize, Ringing below 0 Clipped)
    #   (done in Chunks of Volumes, as a full 96 x 96 x 81 x ~400 Run is ~1.2 GB in float32)
    #   (Run divided by its Maximum first: raw uint16 Intensities reach ~64000, beyond float16's 65504 once resized;
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

    # Task Condition Fetching Function (Per-Volume Label from the BIDS Events File, by Volume Onset Time)
    #   ExperienceSampling: 'probe' during a Trial's Rating Probes (first Item Onset -> last Item Offset), else 'fixation'
    #   GradCPT: 'gradcpt' for every Volume
    #   (some Runs only ship '_events_corrected.tsv' / '_events_og.tsv'; the Probe Timings are the same in both)
    def get_conditions(self, run_id: str, n_vol: int, tr: float):
        _, _, task, _ = self.parse_run(run_id)
        if task != 'ExperienceSampling': return [task.lower()] * n_vol
        prefix = self.run_fp(run_id, 'func')
        tsv = next((Path(f"{prefix}{s}.tsv") for s in ('_events', '_events_corrected', '_events_og')
                    if Path(f"{prefix}{s}.tsv").exists()), None)
        if tsv is None: return ['n/a'] * n_vol
        lines = tsv.read_text().splitlines(); head = lines[0].strip().split('\t')
        c_on, c_off, c_tr = head.index('onset'), head.index('offset'), head.index('trial')
        probes = {}
        for line in lines[1:]:
            cols = line.strip().split('\t')
            if len(cols) != len(head) or cols[c_on] == 'n/a': continue
            on, off = float(cols[c_on]), float(cols[c_off]); lo, hi = probes.get(cols[c_tr], (on, off))
            probes[cols[c_tr]] = (min(lo, on), max(hi, off))
        return ['probe' if any(lo <= v * tr < hi for lo, hi in probes.values()) else 'fixation' for v in range(n_vol)]

    # ============================================================================================

    # Preprocessed Run Filepath Function
    def prep_files(self, run_id: str):
        subj_idx, ses_idx, _, _ = self.parse_run(run_id); prep_dir = self.prep_fp / f"sub-{subj_idx}" / f"ses-{ses_idx}"
        return {k: prep_dir / f"{run_id}_{k}.{'json' if k == 'meta' else 'npy'}" for k in ('eeg', 'fmri', 'meta')}

    # Full Run Preprocessing Function (+ Saving if save_prep)
    def prep_run(self, run_id: str, save: bool = None):
        save = self.args.save_prep if save is None else save
        if self.args.verbose: print(f"Preprocessing Run | {run_id}")
        fmri_meta, fmri_img = self.get_fMRI(run_id)
        n_vol, tr = fmri_img.shape[-1], fmri_meta['RepetitionTime']
        fmri, fmri_scale = self.prep_fMRI(fmri_img); del fmri_img
        eeg, valid, qc = self.prep_EEG(run_id, n_vol, tr)
        meta = {'run_id': run_id, 'n_vol': n_vol, 'tr': tr, 'eeg_shape': list(eeg.shape), 'fmri_shape': list(fmri.shape), 'fmri_scale': fmri_scale,
                'eeg_valid': valid.tolist(), 'condition': self.get_conditions(run_id, n_vol, tr), 'qc': qc}

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
    # (~0.6 MB per run), not the LRU run cache, so shuffled batches never recompute it
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
