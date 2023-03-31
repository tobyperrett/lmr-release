#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:8
#SBATCH --job-name=vtf
#SBATCH --mem=500GB
#SBATCH --nodes=1
#SBATCH --partition=big
#SBATCH --time=24:00:00

source /jmain02/home/J2AD001/wwp01/shared/home/etc/profile
conda activate motionformer

export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
export MASTER_PORT=19500

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo
echo $SLURMD_NODENAME 
echo $SLURM_NODELIST
echo $SLURM_JOB_ID
echo $CUDA_VISIBLE_DEVICES
master_node=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:4}
dist_url="tcp://"
dist_url+=$master_node
dist_url+=:40000
echo $dist_url


if [ -z "$1" ]
then
    CFG='/jmain02/home/J2AD001/wwp01/txp48-wwp01/motionformer/configs/EK/motionformer_224_16x4.yaml'
else
    CFG=$1
fi

if [ -z "$2" ]
then
    ROOT_FOLDER="checkpoint/motionformer"
else
    ROOT_FOLDER=$2
fi

SAV_FOLDER="${ROOT_FOLDER}/${SLURM_JOB_ID}"
mkdir -p ${SAV_FOLDER}
echo SAV_FOLDER
echo $SAV_FOLDER

rm -rf /raid/local_scratch/txp48-wwp01/frames

cp -r /jmain02/home/J2AD001/wwp01/shared/data/epic-100/frames /raid/local_scratch/txp48-wwp01/



for fol in /raid/local_scratch/txp48-wwp01/frames/*; do
   cd $fol
        for f in *.tar; do folder=${f::-4}; mkdir -p $folder; cd $folder; tar -xf ../$f . ; cd ..; done
   cd ..
done

cd /jmain02/home/J2AD001/wwp01/txp48-wwp01/motionformer
python setup.py build develop

export WANDB_MODE=offline

# command
srun --label python tools/run_net.py --init_method $dist_url --num_shards 1 --cfg $CFG \
OUTPUT_DIR ${SAV_FOLDER}