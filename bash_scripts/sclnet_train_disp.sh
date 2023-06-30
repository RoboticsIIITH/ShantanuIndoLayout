#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=sclnet_train_disp
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4-00:00:00
#SBATCH --mincpus=40
#SBATCH --gres=gpu:4
#SBATCH --mail-user=shantanu.singh@research.iiit.ac.in
#SBATCH --mail-type=END

unset http_proxy

username=$USER
rootdir=/scratch/$USER
use_tmpfs=false

rm -rf $rootdir/HabitatGibson
mkdir -p $rootdir
cd $rootdir

wget 'http://datasets.rrc.iiit.ac.in/user-datasets/jaidev/ZipFiles/HabitatGibsonAll.zip'

unzip HabitatGibsonAll.zip

tmp_path=$rootdir/HabitatGibson

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

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate py37

cd $rootdir
cp ~/indoor-layout-estimation.zip .
unzip indoor-layout-estimation.zip
cd indoor-layout-estimation/
cd networks/stereo_depth/anynet/models/spn_t1
bash make.sh
pip install -e .

project_dir=$rootdir/indoor-layout-estimation
cd $project_dir

train_split_file=splits/gibson/gibson_train_depth.txt
val_split_file=splits/gibson/gibson_val_depth.txt
log_dir=$project_dir/results

mkdir -p $log_dir

# md_dgt
CUDA_VISIBLE_DEVICES=0,1 python3 train_posenet_bev.py configs/train_disp.yaml \
    model_name md_dgt \
    data_path $data_path \
    log_dir $log_dir \
    load_weights_folder None \
    depth_dir None \
    mode dense \
    train_split_file $train_split_file val_split_file $val_split_file \
    train_workers 15 val_workers 5 >results/md_dgt.txt 2>&1


cd $log_dir
zip -r md_dgt.zip md_dgt/
