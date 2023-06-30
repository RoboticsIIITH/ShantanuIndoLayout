#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=sclnet_train_bev
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --mincpus=40
#SBATCH --gres=gpu:4
#SBATCH --mail-user=shantanu.singh@research.iiit.ac.in
#SBATCH --mail-type=END

unset http_proxy

username=$USER
use_tmpfs=true

tmp_path=/scratch/$username/gibson_data
rm -rf $tmp_path
mkdir -p $tmp_path
cd $tmp_path

scp shantanu.singh@ada:/share1/shantanu.singh/HabitatGibsonAll.zip .

unzip HabitatGibsonAll.zip

tmp_path=/scratch/$username/gibson_data/HabitatGibson

if [ "$use_tmpfs" = true ] ; then
    tmp_path=/dev/shm/HabitatGibson/
    rm -rf $tmp_path
    mkdir -p $tmp_path
    echo 'Copying data to $tmp_path'
    rsync -av --ignore-errors HabitatGibson/ $tmp_path
fi

data_path=$tmp_path/data

module load cuda/10.2
module load cudnn/7.6.5-cuda-10.2

. /home/$username/miniconda3/etc/profile.d/conda.sh
conda activate py37

# cd /scratch/$username
# git clone https://github.com/jaidevshriram/indoor-layout-estimation.git
# cd indoor-layout-estimation/
# git checkout disp_train
# cd networks/stereo_depth/anynet/models/spn_t1
# bash make.sh
# pip install -e .
# cd /scratch/$username/indoor-layout-estimation

project_dir=~/indoor-layout-estimation
cd $project_dir

train_split_file=splits/gibson/gibson_train_depth.txt
val_split_file=splits/gibson/gibson_val_depth.txt
log_dir=/scratch/$username/indoor-layout-estimation/results

mkdir -p $log_dir

# mb_dgt_bsim
CUDA_VISIBLE_DEVICES=0 python3 train_posenet_bev.py configs/train_bev.yaml \
    model_name mb_dgt_bsim \
    data_path $data_path \
    log_dir $log_dir \
    load_weights_folder None \
    depth_dir None \
    bev_dir /dev/shm/HabitatGibson/bevs/sim \
    mode debug \
    BEV_DECODER.n_channels 4 \
    train_split_file $train_split_file val_split_file $val_split_file \
    train_workers 8 val_workers 2 >results/mb_dgt_bsim.txt 2>&1 &

# mb_-_bsim
CUDA_VISIBLE_DEVICES=1 python3 train_posenet_bev.py configs/train_bev.yaml \
    model_name mb_-_bsim \
    data_path $data_path \
    log_dir $log_dir \
    load_weights_folder None \
    depth_dir None \
    bev_dir /dev/shm/HabitatGibson/bevs/sim \
    mode debug \
    BEV_DECODER.n_channels 3 \
    train_split_file $train_split_file val_split_file $val_split_file \
    train_workers 8 val_workers 2 >results/mb_-_bsim.txt 2>&1 &

# mb_dgt_bgtd
CUDA_VISIBLE_DEVICES=2 python3 train_posenet_bev.py configs/train_bev.yaml \
    model_name mb_dgt_bgtd \
    data_path $data_path \
    log_dir $log_dir \
    load_weights_folder None \
    depth_dir None \
    bev_dir /dev/shm/HabitatGibson/bevs/gtdepth \
    mode debug \
    BEV_DECODER.n_channels 4 \
    train_split_file $train_split_file val_split_file $val_split_file \
    train_workers 8 val_workers 2 >results/mb_dgt_bgtd.txt 2>&1 &

# mb_sgbm_dpred_bpred
CUDA_VISIBLE_DEVICES=3 python3 train_posenet_bev.py configs/train_bev.yaml \
    model_name mb_sgbm_dpred_bpred \
    data_path $data_path \
    log_dir $log_dir \
    load_weights_folder None \
    depth_dir /dev/shm/HabitatGibson/sgbm_preddepth \
    bev_dir /dev/shm/HabitatGibson/bevs/preddepth \
    mode debug \
    BEV_DECODER.n_channels 4 \
    train_split_file $train_split_file val_split_file $val_split_file \
    train_workers 8 val_workers 2 >results/mb_sgbm_dpred_bpred.txt 2>&1


cd /scratch/$username/indoor-layout-estimation
zip -r train_bev.zip results/
scp train_bev.zip $username@ada:/share1/$username/
