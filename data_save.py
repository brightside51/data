# Package Imports
import sys
import os
import argparse
import torch
#import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================================

# Non-Conditional 3D Diffusion Model Parser Initialization
if True:
    ncdiff_parser = argparse.ArgumentParser(
        description = "Non-Conditional 3D Diffusion Model")
    ncdiff_parser.add_argument('--model_type', type = str,            # Chosen Model / Diffusion
                                choices =  {'video_diffusion',
                                            'blackout_diffusion',
                                            'gamma_diffusion'},
                                default = 'video_diffusion')
    ncdiff_parser.add_argument('--model_version', type = int,         # Model Version Index
                                default = 0)
    ncdiff_parser.add_argument('--data_version', type = int,          # Dataset Version Index
                                default = 5)
    ncdiff_parser.add_argument('--noise_type', type = str,            # Diffusion Noise Distribution
                                default = 'gaussian')
    settings = ncdiff_parser.parse_args("")

    # ============================================================================================

    # Directories and Path Arguments
    ncdiff_parser.add_argument('--reader_folderpath', type = str,         # Path for Dataset Reader Directory
                                default = '../../data/non_cond')
    ncdiff_parser.add_argument('--public_data_folderpath', type = str,    # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
                                default = "../../../../../datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
    ncdiff_parser.add_argument('--private_data_folderpath', type = str,   # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/METABREST/T1W_Breast")
                                default = '../../../../../datasets/private/METABREST/T1W_Breast')
    ncdiff_parser.add_argument( '--lung_data_folderpath', type = str,     # Path for LUCAS Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")
                                default = "../../../../../datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")

    # Directory | Model-Related Path Arguments
    ncdiff_parser.add_argument('--model_folderpath', type = str,          # Path for Model Architecture Directory
                                default = f'../../models/{settings.model_type}')
    ncdiff_parser.add_argument('--script_folderpath', type = str,         # Path for Model Training & Testing Scripts Directory
                                default = f'../../scripts/{settings.model_type}')
    ncdiff_parser.add_argument('--logs_folderpath', type = str,           # Path for Model Saving Directory
                                default = f'../../logs/{settings.model_type}')
        
    # ============================================================================================

    # Dataset | Dataset General Arguments
    ncdiff_parser.add_argument('--data_format', type = str,           # Chosen Dataset Format for Reading
                                choices =  {'mp4', 'dicom'},
                                default = 'dicom')
    ncdiff_parser.add_argument('--img_size', type = int,              # Generated Image Resolution
                                default = 512)
    ncdiff_parser.add_argument('--num_slice', type = int,             # Number of 2D Slices in MRI
                                default = 30)
    ncdiff_parser.add_argument('--slice_spacing', type = bool,        # Usage of Linspace for Slice Spacing
                                default = True)
    ncdiff_parser.add_argument('--slice_bottom_margin', type = int,   # Number of 2D Slices to be Discarded in Bottom Margin
                                default = 5)
    ncdiff_parser.add_argument('--slice_top_margin', type = int,      # Number of 2D Slices to be Discarded in Top Margin
                                default = 15)
    ncdiff_parser.add_argument('--data_prep', type = bool,            # Usage of Dataset Pre-Processing Control Value
                                default = True)
    ncdiff_parser.add_argument('--h_flip', type = int,                # Percentage of Horizontally Flipped Subjects
                                default = 50)

    # Dataset | Dataset Splitting Arguments
    ncdiff_parser.add_argument('--train_subj', type = int,            # Number of Random Subjects in Training Set
                                default = 0)                          # PS: Input 0 for all Subjects in the Dataset
    ncdiff_parser.add_argument('--val_subj', type = int,              # Number of Random Subjects in Validation Set
                                default = 0)
    ncdiff_parser.add_argument('--test_subj', type = int,             # Number of Random Subjects in Test Set
                                default = 0)

    # Dataset | DataLoader Arguments
    ncdiff_parser.add_argument('--batch_size', type = int,            # DataLoader Batch Size Value
                                default = 1)
    ncdiff_parser.add_argument('--shuffle', type = bool,              # DataLoader Subject Shuffling Control Value
                                default = True)
    ncdiff_parser.add_argument('--num_workers', type = int,           # Number of DataLoader Workers
                                default = 8)
    ncdiff_parser.add_argument('--num_fps', type = int,               # Number of Video Frames per Second
                                default = 4)

    # ============================================================================================

    # Model | Architecture-Defining Arguments
    ncdiff_parser.add_argument('--seed', type = int,                  # Randomised Generational Seed
                                default = 0)
    ncdiff_parser.add_argument('--dim', type = int,                   # Input Dimensionality (Not Necessary)
                                default = 64)
    ncdiff_parser.add_argument('--num_channel', type = int,           # Number of Input Channels for Dataset
                                default = 1)
    ncdiff_parser.add_argument('--mult_dim', type = tuple,            # Dimensionality for all Conditional Layers
                                default = (1, 2, 4, 8))

    # Model | Training & Diffusion Arguments
    #ncdiff_parser.add_argument('--num_epochs', type = int,            # Number of Training Epochs
    #                            default = 30)
    ncdiff_parser.add_argument('--num_ts', type = int,                # Number of Scheduler Timesteps
                                default = 300)
    ncdiff_parser.add_argument('--num_steps', type = int,             # Number of Diffusion Training Steps
                                default = 500000)
    ncdiff_parser.add_argument('--lr_base', type = float,             # Base Learning Rate Value
                                default = 1e-4)
    ncdiff_parser.add_argument('--save_interval', type = int,         # Number of Training Step Interval inbetween Image Saving
                                default = 1000)
    ncdiff_parser.add_argument('--save_img', type = int,              # Square Root of Number of Images Saved for Manual Evaluation
                                default = 2)

    # ============================================================================================

    settings = ncdiff_parser.parse_args("")
    settings.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# Functionality Imports
sys.path.append(settings.reader_folderpath)
from nc_data_reader import NCDataset

# ============================================================================================

# Dataset Saving Example
print("Starting")
public_dataset = NCDataset( settings,
                            mode = 'train',
                            dataset = 'public')
print(len(public_dataset))
private_dataset = NCDataset(settings,
                            mode = 'train',
                            dataset = 'private')
print(len(private_dataset))
for i in range(0, len(public_dataset)):
    data = public_dataset.__getitem__(i, save = True)
    print(f"{i} -> {data.shape}\n")
for i in range(0, len(private_dataset)):
    data = private_dataset.__getitem__(i, save = True)
    print(f"{i} -> {data.shape}\n")

#print(data.shape); print(torch.max(data)); print(torch.min(data))
#subj_filelist = os.listdir(Path("X:/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI/video_data/V3/train"))
#for i in range(len(subj_filelist), len(dataset)): dataset.__getitem__(i, save = True)
