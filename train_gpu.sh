#!/bin/bash
#srun --mem 400G --gres=gpu --partition=amd-longq --nodes 1 sh train.sh
sbatch -o train.%J.log -J XCVAE --mem 32G --partition intel-longq --gres=gpu --nodelist=dgx01 train.sh
