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
    assert dataset in [None, 'metabreast', 'dukebreast',
                       'lidc_idri', 'luna25_nodule', 'covid_jun2020',
                       'ds003768', 'noddi'],\
        "ERROR |ARGUMENT PARSER cannot be initialised with the chosen Dataset"
    parser.add_argument('--dataset', type = str,
                        choices = { None, 'metabreast', 'dukebreast',
                                    'lidc_idri', 'luna25_nodule', 'covid_jun2020',
                                    'ds003768', 'noddi'},
                        default = dataset)

    # Dataset Versioning
    parser.add_argument('--dataV', type = str, default = dataV)
    parser.add_argument('--verbose', type = bool, default = True)
    if 'deucalion' in dataV or 'hpc' in dataV:
        parser.add_argument('--base_fp', type = str,
            default = "V:/projects/F202509580CPCAA1/pedrofsousa/data")
    else:
        parser.add_argument('--base_fp', type = str,
            default = "/nas-ctm01/homes/pfsousa/data")
    args = parser.parse_args("")

    # --------------------------------------------------------------------------------------------

    # Load Existing Arguments if Available
    save_fp = Path(f"{args.base_fp}/{args.dataset}/{args.dataV}_args.yaml")
    print(save_fp)
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
        if 'deucalion' in dataV or 'hpc' in dataV:
            parser.add_argument('--data_fp', type = str,
                                default = f"/projects/F202509580CPCAA1/datasets/{args.dataset}")
        else:
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
            elif args.dataset == 'luna25_nodule':
                parser.add_argument('--data_fp', type = str,
                                    default = "/nas-ctm01/datasets/public/medical_datasets/lung_ct_datasets/luna25_challenge/data_staging_area_II/protocol_1/data_npy_files/lung_nodule")
            elif args.dataset == 'covid_jun2020':
                parser.add_argument('--data_fp', type = str,
                                    default = "/nas-ctm01/datasets/public/covid_jun2020")
            elif args.dataset == 'ds003768':
                            parser.add_argument('--data_fp', type = str,
                                                default = "/nas-ctm01/datasets/public/ds003768/ds003768")
            
        # ============================================================================================

        # Argument File Saving
        args = parser.parse_args("")
        if save:
            if args.verbose: print(f"Saving ARGUMENT PARSER | {save_fp}")
            if not save_fp.parent.exists(): os.makedirs(save_fp.parent)
            with open(save_fp, "w") as f: yaml.safe_dump(vars(args), f, sort_keys = False)
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    return args

# ============================================================================================