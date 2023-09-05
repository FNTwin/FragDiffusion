#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=fragdiff

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=out_files/fragdiff.out
#SBATCH --error=out_files/fragdiff.err
## SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=7-00:00:00

## How many CPUs to request. Maximum is 124.
#SBATCH --cpus-per-task=4

## Rerun on walltime
##SBATCH --signal=SIGHUP@90

## How much memory to request in MB. Maximum is 460GB.
#SBATCH --mem=40GB

## Partition to use,
#SBATCH --partition=v1002 #a1001
##SBATCH --nodelist=nodes[30,31]
## --nodes=2
## Submit an array of jobs
##SBATCH --array=0-5

## How many GPU to request. Maximum is 3.
#SBATCH --gres=gpu:1
## SBATCH --ntasks-per-node=2
## You can also request a percentage of one GPU.
## Example to get 20% of a GPU.
## This approach has severe limitation as it can only be used by
## a single user at a time. More tests should be performed whether this
## ok to use it. Please experiment and report findings.
## #SBATCH --gres=mps:50

set -e

# The below env variables can eventually help setting up your workload.
echo "SLURM_JOB_UID=$SLURM_JOB_UID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_COUNT=$SLURM_ARRAY_TASK_COUNT"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

# In a SLURM job, you CANNOT use `conda activate` and instead MUST use:
# cd ~/local/conda/bin
source activate fragdiff

# Execute your workload
# cd /home/nikhil_valencediscovery_com/projects/openMLIP
# srun python src/mlip/train.py experiment=drugs-chem-hyp-v2/mace-small/1m-n32.yaml

# A dummy and useless `sleep` to give you time to see your job with `squeue`.
# select data where hydra config wills search
#export DATA_DIR="/home/cristian_valencediscovery_com/dev/openMLIP/expts/"
export HYDRA_FULL_ERROR=1
# tiny variant 200k
#python ../src/mlip/train.py experiment=spice-v2.0-with-forces/mace-opt-tiny.yaml
# small variant 2M
srun python ../main.py dataset=frag
# base variant 20M
#python ../src/mlip/train.py experiment=spice-v2.0-with-forces/mace-opt-base.yaml

#sleep 20s
