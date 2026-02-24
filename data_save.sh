#!/bin/bash
#
#SBATCH --partition=gpu_min24gb                                     # Partition (check with "$sinfo")
#SBATCH --output=output.out                                         # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=error.err                                           # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=metabrest                                        # (Optional) Job name
#SBATCH --time=14-00:00                                             # (Optional) Time limit (D: days, HH: hours, MM: minutes)
#SBATCH --qos=gpu_min24GB                                           # (Optional) 01.ctm-deep-05

#Commands / scripts to run (e.g., python3 train.py)
python data_save.py