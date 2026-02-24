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
            self.full_idx = os.listdir(self.args.data_fp); self.full_idx.remove('video_data'); self.get_split()
        
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

        # DICOM Subject File Reading
        if self.args.data_format == 'dicom':
            print(True)

            # Subject Folder Access
            idx_fop = f"{self.args.data_fp}/{idx}"
            idx_flist = os.listdir(idx_fop)
            for i, path in enumerate(idx_flist):
                idx_fop = f"{self.args.data_fp}/{idx}/{path}"
                idx_flist = os.listdir(idx_fop)
                while os.path.splitext(idx_flist[0])[1] not in ['.dcm', '.xlm']:
                    idx_fop = Path(f"{idx_fop}/{idx_flist[0]}")
                    idx_flist = os.listdir(idx_fop)
                if len(idx_flist) >= 50: break
            idx_flist = np.ndarray.tolist(np.sort(idx_flist))
            
            # Subject General Information Access
            idx_fp = Path((f"{idx_fop}/{idx_flist[0]}"))
            while os.path.splitext(idx_fp)[1] not in ['', '.dcm']:
                i += 1; idx_fp = Path((f"{idx_fop}/{idx_flist[i]}"))
            idx_info = pydicom.dcmread(idx_fp, force = True)
            idx_og = int(idx_info[0x0020, 0x0013].value)
            idx_ori = idx_info[0x0020, 0x0037].value
            idx_vflip = (np.all(idx_ori == [-1, 0, 0, 0, -1, 0]))
            idx_hflip = (torch.rand(1) < (self.args.h_flip / 100))

            # --------------------------------------------------------------------------------------------
                
            # Subject Slice Data Access
            frame_og = len(idx_flist) + idx_og + 100
            img_data = torch.empty((frame_og, self.args.img_size, self.args.img_size)); slice_list = []
            for i, slice_fp in enumerate(np.sort(idx_flist)):
                if os.path.splitext(slice_fp)[1] in ['', '.dcm']:
                    
                    # Slice Data Access
                    slice_fp = Path(f"{idx_fop}/{slice_fp}")
                    slice_info = pydicom.dcmread(slice_fp, force = True)
                    slice_idx = int(slice_info[0x0020, 0x0013].value)
                    if slice_idx >= len(idx_flist) + 1: slice_idx -= idx_og 
                    slice_list.append(slice_idx)
                    slice_data = slice_info.pixel_array.astype(float)

                    # Slice Image Pre-Processing | Rescaling, Resizing & Flipping
                    if self.args.data_prep:
                        slice_data = np.uint8((np.maximum(slice_data, 0) / slice_data.max()) * 255)
                        slice_data = Image.fromarray(slice_data).resize((   self.args.img_size,
                                                                            self.args.img_size)) 
                        if idx_hflip: slice_data = self.h_flip(slice_data)
                        if idx_vflip: slice_data = self.v_flip(slice_data)
                        slice_data = np.array(self.transform(slice_data))
                    img_data[slice_idx, :, :] = torch.Tensor(slice_data); del slice_data
                else: idx_flist.remove(slice_fp)
            print(f"Accessing Subject {idx}: {len(idx_flist)} -> {self.args.num_slice} Slices")
            img_data = img_data[np.sort(slice_list)]

            # --------------------------------------------------------------------------------------------

            # Slice Cropping | Spaced-Out Slices
            if self.args.slice_spacing:
                s_array = slice_array = np.linspace(0 + self.args.slice_bottom_margin,
                    len(idx_flist) - self.args.slice_top_margin - 1, self.args.num_slice)
                slice_array[0 : int(np.floor(self.args.num_slice / 2))] = np.ceil(s_array[0 : int(np.floor(self.args.num_slice / 2))]).astype(int)
                slice_array[int(np.ceil(self.args.num_slice / 2) + 1)::] = np.floor(s_array[int(np.ceil(self.args.num_slice / 2) + 1)::]).astype(int)
                slice_array[int(np.floor(self.args.num_slice / 2))] = np.round(s_array[int(np.floor(self.args.num_slice / 2))])
                img_data = img_data[slice_array, :, :]

            # Slice Cropping | Middle Slices Only
            else:
                extra_slice = self.args.num_slice - img_data.shape[0]
                if img_data.shape[0] < self.args.num_slice:             # Addition of Repeated Peripheral Slices
                    for extra in range(extra_slice):
                        if extra % 2 == 0: img_data = torch.cat((img_data, img_data[-1].unsqueeze(0)), dim = 0)
                        else: img_data = torch.cat((img_data[0].unsqueeze(0), img_data), dim = 0)
                elif img_data.shape[0] > self.args.num_slice:           # Removal of Emptier Peripheral Slices
                    img_data = img_data[int(np.ceil(-extra_slice / 2)) :\
                        int(len(img_data) - np.floor(-extra_slice / 2))]
            assert(img_data.shape[0] == self.args.num_slice)
          
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