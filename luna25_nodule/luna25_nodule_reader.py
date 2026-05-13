# Library Imports
import os
import random
import argparse
import numpy as np
import pydicom
import torch
import torchvision
import matplotlib.pyplot as plt

# Function Imports
from pathlib import Path
from ipywidgets import interactive, IntSlider
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ============================================================================================

# Non-Conditional 3D Medical Imaging Dataset Reader Class
class NCDataset(Dataset):

    # Constructor / Initialisation
    def __init__(
        self,
        args: argparse.ArgumentParser,
        mode: str = 'train',
    ):  
        
        # Argument Initialisation
        super(NCDataset).__init__()
        self.args = args; self.mode = mode

        # Data Split Finding / Creation
        split_fp = Path(f"{self.args.reader_fp}/{self.args.dataV}_{self.mode}.txt")
        if self.args.verbose: print(f"Split Filepath | {split_fp}")
        if split_fp.exists():
            if self.args.verbose: print(f"Reading Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.full_idx = split_fp.read_text().splitlines()
        else:
            if self.args.verbose: print(f"Creating Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.full_idx = os.listdir(self.args.data_fp)
            if 'video_data' in self.full_idx: self.full_idx.remove('video_data')
            self.get_split()
        
        # --------------------------------------------------------------------------------------------
        
        # Dataset Transformations Initialisation
        if self.args.img_size != 0:
            self.transform =    transforms.Compose([
                                transforms.Resize(( self.args.img_size,
                                                    self.args.img_size)),
                                transforms.ToTensor()])
        else:
                self.transform = transforms.Compose([transforms.ToTensor()])
        self.h_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1)])
        self.v_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1)])

    # ============================================================================================

    # DataLoader Length / No. Subjects Computation
    def __len__(self): return len(self.full_idx)

    # --------------------------------------------------------------------------------------------

    # Data Splitting
    def get_split(self):

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

   # Single Batch Fetching
    def __getitem__(self, idx: int = 0 or str, save: bool = False):
        if type(idx) == int: idx = self.full_idx[idx]
        if self.args.verbose: print(f"Fetching Data  | Patient ID {idx} | {self.args.data_format} Format")

        # NPY Subject File Reading
        if self.args.data_format == 'npy':
            full_data = torch.Tensor(np.load(f"{self.args.data_fp}/{idx}"))
            img_size = full_data.shape[-1] if self.args.img_size == 0 else self.args.img_size
            img_data = torch.empty((full_data.shape[0], img_size, img_size))

            # Slice Image Pre-Processing | Rescaling, Resizing & Flipping
            idx_hflip = (torch.rand(1) < (self.args.h_flip / 100))
            if self.args.data_prep:
                #full_data = np.uint8((np.maximum(full_data, 0) / full_data.max()) * 255)
                full_data = np.array(full_data)
                for s in range(full_data.shape[0]):
                    slice_data = Image.fromarray(full_data[s]).resize(( img_size, img_size))
                    if idx_hflip: slice_data = self.h_flip(slice_data)
                    slice_data = torch.Tensor(self.transform(slice_data))
                    img_data[s, :, :] = torch.rot90(slice_data, k = -1, dims = (-2, -1)); del slice_data

            # --------------------------------------------------------------------------------------------
          
            # Sample Saving | MP4 Format
            if save:
                print(f"Saving Data | Patient ID {idx} | MP4 Format")
                if not os.path.isdir(f"{self.args.data_fp}/video_data/V{self.args.data_version}"):
                    os.mkdir(f"{self.args.data_fp}/video_data/V{self.args.data_version}")
                if not os.path.isdir(f"{self.args.data_fp}/video_data/V{self.args.data_version}/{self.mode}"):
                    os.mkdir(f"{self.args.data_fp}/video_data/V{self.args.data_version}/{self.mode}")
                torchvision.io.write_video(f"{self.args.data_fp}/video_data/V{self.args.data_version}/{self.mode}/{idx}.mp4",
                    (img_data.unsqueeze(3).repeat(1, 1, 1, 3) * 255).type(torch.uint8), fps = self.args.num_fps)
            
            # Sample Saving | Torch Format
            if save:
                print(f"Saving Data | Patient ID {idx} | Torch Format")
                

        # --------------------------------------------------------------------------------------------
        
        # MP4 Subject File Reading
        elif self.args.data_format == 'mp4':

            # Subject Data Access
            idx_fp = f"{self.args.data_fp}/video_data/{self.args.num_slice}x{self.args.img_size}x{self.args.img_size}/{idx}.mp4"
            print(f"Subject Filepath | {idx_fp}")
            img_data = (torchvision.io.read_video(idx_fp, pts_unit = 'sec')[0][:, :, :, 0] / 255.0).type(torch.float32)

        else: raise(NotImplementedError)
        return img_data.unsqueeze(0)
    
    # ============================================================================================