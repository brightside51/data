# Library Imports
import os
import random
import argparse
import numpy as np
import pydicom
import torch
import torchvision
import matplotlib.pyplot as plt
import nibabel as nib
import cv2

# Function Imports
from pathlib import Path
from ipywidgets import interactive, IntSlider
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from itertools import chain  

# ============================================================================================

# Non-Conditional 2D Medical Imaging Dataset Reader Class
class DataReader(Dataset):

    # Constructor / Initialisation
    def __init__(
        self,
        args: argparse.ArgumentParser,
        mode: str = 'train',
    ):  
        
        # Argument Initialisation
        super(DataReader).__init__()
        self.args = args; self.mode = mode

        # Data Split Finding / Creation
        split_fp = Path(f"{self.args.reader_fp}/{self.args.dataV}_{self.mode}.txt")
        if self.args.verbose: print(f"Split Filepath | {split_fp}")
        if split_fp.exists():
            if self.args.verbose: print(f"Reading Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.full_idx = split_fp.read_text().splitlines()
        else:
            if self.args.verbose: print(f"Creating Splits | {self.args.dataset} Dataset | {self.mode} Set | Version {args.dataV}")
            self.full_idx = os.listdir(f"{self.args.data_fp}/ct")
            if 'ReadMe' in self.full_idx: self.full_idx.remove('ReadMe')
            self.get_split()

        # Current Subject Initialisation
        if self.args.subj_shuffle: random.shuffle(self.full_idx)
        self.curr_subj_idx = 0; self.num_slice = self.__len__()
        self.curr_subj_ct, self.curr_subj_mask = self.getsubj(self.curr_subj_idx)
        
        # --------------------------------------------------------------------------------------------
        
        # Dataset Transformations Initialisation
        self.transform =    transforms.Compose([
                            transforms.Resize(( self.args.img_size,
                                                self.args.img_size)),
                            transforms.ToTensor()])
        self.h_flip = transforms.Compose([transforms.RandomHorizontalFlip(p = 1)])
        self.v_flip = transforms.Compose([transforms.RandomVerticalFlip(p = 1)])

    # ============================================================================================

    # DataLoader Length / No. Subjects Computation
    def __len__(self):
        
        try: return self.num_slice
        except AttributeError:
            self.subj_list, self.slice_list = [], []
            for subj in range(0, len(self.full_idx)):
                subj_ct, subj_mask = self.getsubj(subj)
                subj_slice = list(range(0, subj_ct.shape[0]))
                if self.args.slice_shuffle: random.shuffle(subj_slice)
                self.subj_list.append([self.full_idx[subj]] * subj_ct.shape[0])
                self.slice_list.append(subj_slice)
                del subj_ct, subj_mask, subj_slice
            self.subj_list = list(chain(*self.subj_list))
            self.slice_list = list(chain(*self.slice_list))
            return len(self.slice_list)

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

    # 2D CT Slice Pre-Processing
    def prep_ct_slice(self, img):
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_min < -500 or img_max > 500:
            img = np.clip(img, self.args.hu_range[0], self.args.hu_range[1])
            img = (img - self.args.hu_range[0]) / (self.args.hu_range[1] - self.args.hu_range[0] + 1e-8)
        else:
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = cv2.resize(   img, (self.args.img_size, self.args.img_size),
                            interpolation = cv2.INTER_LINEAR)
        return img.astype(np.float32)

    def prep_mask_slice(self, mask):
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.resize(  mask, (self.args.img_size, self.args.img_size),
                            interpolation = cv2.INTER_NEAREST)
        return mask.astype(np.uint8)
    
    # ============================================================================================

    def __getitem__(self, idx: int = 0):
        
        # Current Subject Update
        slice_idx = self.slice_list[idx]
        if self.subj_list[idx] != self.full_idx[self.curr_subj_idx]:
            self.curr_subj_idx = self.full_idx.index(self.subj_list[idx])
            self.curr_subj_ct, self.curr_subj_mask = self.getsubj(self.curr_subj_idx)

        # Current Slice Fetching
        if self.args.verbose:
            print(f"Subject #{self.curr_subj_idx + 1}/{len(self.full_idx)} | {self.full_idx[self.curr_subj_idx]}")
            #print(f"Slice #{idx + 1}/{self.num_slice} (Total) | {slice_idx + 1}/{self.curr_subj_ct.shape[0]} (Current Subject)")
        ct_slice = self.curr_subj_ct[slice_idx, :, :]
        mask_slice = self.curr_subj_mask[slice_idx, :, :]
        
        return ct_slice.unsqueeze(0), mask_slice.unsqueeze(0)
    
    # ============================================================================================

    # Single Batch Fetching
    def getsubj(self, idx: int = 0 or str, save: bool = False):
        if type(idx) == int: idx = self.full_idx[idx]
        if self.args.verbose: print(f"Fetching Data  | Patient ID {idx} | {self.args.data_format} Format")

        # NIFTI Subject File Reading
        if self.args.data_format == 'npy':
            ct_data = np.load(f"{self.args.data_fp}/ct/{idx}")
            mask_data = np.load(f"{self.args.data_fp}/lungmask/{idx}")

            # Slice Image Pre-Processing | HU Clipping, flipping, Rescaling & Resizing
            idx_hflip = (torch.rand(1) < (self.args.h_flip / 100)); ct, mask = [], []
            for s in range(ct_data.shape[0]):
                ct_slice = self.prep_ct_slice(ct_data[s, :, :])
                mask_slice = self.prep_mask_slice(mask_data[s, :, :])

                # Slice Filtering | Mask Area Thresholding
                #if idx_hflip: ct_slice = self.h_flip(ct_slice)
                if np.sum(mask_slice > 0) >= self.args.min_mask_area:
                    ct.append(ct_slice); mask.append(mask_slice)

            ct = torch.Tensor(np.array(ct, dtype = np.float32))
            mask = torch.Tensor(np.array(mask, dtype = np.uint8))

            # --------------------------------------------------------------------------------------------

            # Slice Cropping | Spaced-Out Slices
            if self.args.num_slice != 0:
                              
                # Slice Cropping | Spaced-Out Slices
                if self.args.slice_spacing:
                    s_array = slice_array = np.linspace(0 + self.args.slice_bottom_margin,
                        len(idx_flist) - self.args.slice_top_margin - 1, self.args.num_slice)
                    slice_array[0 : int(np.floor(self.args.num_slice / 2))] = np.ceil(s_array[0 : int(np.floor(self.args.num_slice / 2))]).astype(int)
                    slice_array[int(np.ceil(self.args.num_slice / 2) + 1)::] = np.floor(s_array[int(np.ceil(self.args.num_slice / 2) + 1)::]).astype(int)
                    slice_array[int(np.floor(self.args.num_slice / 2))] = np.round(s_array[int(np.floor(self.args.num_slice / 2))])
                    ct = ct[slice_array, :, :]; mask = mask[slice_array, :, :]

                # Slice Cropping | Middle Slices Only
                else:
                    extra_slice = self.args.num_slice - ct.shape[0]
                    if ct.shape[0] < self.args.num_slice:             # Addition of Repeated Peripheral Slices
                        for extra in range(extra_slice):
                            if extra % 2 == 0:
                                ct = torch.cat((ct, ct[-1].unsqueeze(0)), dim = 0)
                                mask = torch.cat((mask, mask[-1].unsqueeze(0)), dim = 0)
                            else:
                                ct = torch.cat((ct[0].unsqueeze(0), ct), dim = 0)
                                mask = torch.cat((mask[0].unsqueeze(0), mask), dim = 0)
                    elif ct.shape[0] > self.args.num_slice:           # Removal of Emptier Peripheral Slices
                        ct = ct[int(np.ceil(-extra_slice / 2)) :\
                            int(len(ct) - np.floor(-extra_slice / 2))]
                        mask = mask[int(np.ceil(-extra_slice / 2)) :\
                            int(len(mask) - np.floor(-extra_slice / 2))]
                assert(ct.shape[0] == self.args.num_slice)
            
            # --------------------------------------------------------------------------------------------

            # Sample Saving | MP4 Format
            if save:
                print(f"Saving Data | Patient ID {idx} | MP4 Format")
                if not os.path.isdir(f"{self.args.data_fp}/video_data/{self.args.dataV}"):
                    os.mkdir(f"{self.args.data_fp}/video_data/{self.args.dataV}")
                torchvision.io.write_video(f"{self.args.data_fp}/video_data/{self.args.dataV}/ct_{idx}.mp4",
                    (ct.unsqueeze(3).repeat(1, 1, 1, 3) * 255).type(torch.uint8), fps = self.args.num_fps)
                torchvision.io.write_video(f"{self.args.data_fp}/video_data/{self.args.dataV}/mask_{idx}.mp4",
                        (mask.unsqueeze(3).repeat(1, 1, 1, 3) * 255).type(torch.uint8), fps = self.args.num_fps)
                
        # --------------------------------------------------------------------------------------------

        else: raise(NotImplementedError)
        return ct, mask
    
    # ============================================================================================