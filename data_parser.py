# Library Imports
import os
import random
import json
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

# Function Imports
from pathlib import Path

# ============================================================================================

def nest_args(flat_dict):
    nested = {}
    for key, value in flat_dict.items():
        if "." in key:
            group, subkey = key.split(".", 1)
            nested.setdefault(group, {})[subkey] = value
        else:
            nested[key] = value
    return nested

def dict_to_namespace(d):
    from argparse import Namespace
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return Namespace(**d)

# --------------------------------------------------------------------------------------------

# Data Arguments Initialisation
def data_parser(
    dataset: str = 'metabreast',
    dataV: str = 'V0',
    save: bool = False,
):  
    
    # Dataset Fundamentals
    parser = argparse.ArgumentParser()
    assert dataset in [None, 'metabreast', 'dukebreast', 'lidc'],\
        "ERROR |ARGUMENT PARSER cannot be initialised with the chosen Dataset"
    parser.add_argument('--dataset', type = str,
                        choices = {None, 'metabreast', 'dukebreast', 'lidc'},
                        default = dataset)
    parser.add_argument('--dataV', type = str, default = dataV)
    parser.add_argument('--verbose', type = bool, default = True)
    parser.add_argument('--base_fp', type = str,
                        default = f"/nas-ctm01/homes/pfsousa/data")
    args = parser.parse_args("")

    # --------------------------------------------------------------------------------------------

    # Load Existing Arguments if Available
    save_fp = Path(f"{args.base_fp}/{args.dataset}/{args.dataV}_args.yaml")
    #save_fp = Path(f"{args.base_fp}/{args.dataset}/{args.dataV}_args.json")
    if save_fp.exists():
        if args.verbose: print(f"Loading ARGUMENT PARSER | {save_fp}")
        with open(save_fp, "r") as f: args = dict_to_namespace(yaml.safe_load(f))
        #with open(save_fp, "r") as f: args = json.load(f)
        #args = argparse.Namespace(**args)
    else:

    # ============================================================================================

        # Directory Arguments
        parser.add_argument('--reader_fp', type = str,
                        default = f"{args.base_fp}/{args.dataset}")
        if args.dataset == 'metabreast':
            parser.add_argument('--data_fp', type = str,
                                default = "/nas-ctm01/datasets/private/METABREST/T1W_Breast")
        elif args.dataset == 'dukebreast':
            parser.add_argument('--data_fp', type = str,
                                default = "/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
        elif args.dataset == 'lidc':
            parser.add_argument('--data_fp', type = str,
                                #default = "/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")
                                default = "/nas-ctm01/datasets/public/lidc_npy/3d/ct")
            
        # ============================================================================================

        # Fundamental Arguments
        parser.add_argument('--data_format', type = str,
                            choices =  {'mp4', 'dicom', 'torch', 'np'},
                            default = 'dicom')
        parser.add_argument('--img_size', type = int, default = 128)
        parser.add_argument('--img_channel', type = int, default = 1)
        parser.add_argument('--num_slice', type = int, default = 30)
        parser.add_argument('--slice_spacing', type = bool, default = False)
        parser.add_argument('--slice_bottom_margin', type = int, default = 5)
        parser.add_argument('--slice_top_margin', type = int, default = 15)
        parser.add_argument('--data_prep', type = bool, default = True)
        parser.add_argument('--h_flip', type = int, default = 50)

        # --------------------------------------------------------------------------------------------

        # Data Splitting Arguments
        parser.add_argument('--val_split', type = int, default = 0)
        parser.add_argument('--test_split', type = int, default = 0)

        # DataLoader Arguments
        parser.add_argument('--batch_size', type = int, default = 1)
        parser.add_argument('--num_fps', type = int, default = 4)
        parser.add_argument('--shuffle', type = bool, default = True)
        parser.add_argument('--num_workers', type = int, default = 12)
        parser.add_argument('--prefetch_factor', type = int, default = 1)

        # ============================================================================================

        # Argument File Saving
        args = parser.parse_args("")
        if save:
            if args.verbose: print(f"Saving ARGUMENT PARSER | {save_fp}")
            if not save_fp.parent.exists(): os.makedirs(save_fp.parent)
            #with open(save_fp, "w") as f: json.dump(vars(args), f)
            with open(save_fp, "w") as f: yaml.safe_dump(vars(args), f)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    return args

# ============================================================================================