#!/bin/bash

#SBATCH --output="train.out"
#SBATCH --partition=gpu
#SBATCH --ntasks=6
#SBATCH --gpus-per-task=3
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

module purge
module load anaconda
source activate cnn

export MASTER_ADDR=c1.cluster
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOB_ID | tail -c 4))

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=1 --node_rank=$SLURM_PROCID --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT code/image_classification.py

