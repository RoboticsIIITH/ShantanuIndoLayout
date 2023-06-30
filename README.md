# Indolayout

## Baselines - Occupancy Anticipation and Active Neural Slam (ANS)

### Training Setup

Please refer to the bash_scripts folder to see examples on how to train or evaluate the Indolayout model on different datasets.

To run the code, you need to setup environment variables, which can be done by sourcing the 'vars' file in bash_scripts folders. 

<b>Pattern:</b>

source bash_scripts/gibson4_exp.vars {Dataset_dir} {Bev_dir} {Log_Dir} {Train_split_path} {Val_split_path}


<b>E.g.:</b>

source bash_scripts/gibson4_exp.vars /home/shantanu/gibson4_dataset /home/shantanu/gibson4_dataset/bev_dir /home/shantanu/indolayout_logs /home/shantanu/indoor-layout-estimation-main/splits/gibson4/filtered_front_train_files.txt /home/shantanu/indoor-layout-estimation-main/splits/gibson4/filtered_front_val_files.txt


### Training ANS RGB or Occant RGB/RGB-D
The main script for training the baseline models (Occupancy Anticipation RGB and RGB-D) is:

train_posenet_bev.py train_occant_gibson4.yaml --script_mode train


### Evaluate ANS RGB or Occant RGB/RGB-D
To evaluate the same model, simply change the script_mode to 'val' as follows:

train_posenet_bev.py train_occant_gibson4.yaml --script_mode val


## Indolayout Model

<b>Parent-dir:</b> cross-view

### Training Setup

Similar to the baselines, to run the code, you need to setup environment variables, which can be done by sourcing the 'vars' file in bash_scripts folders. 

<b>Pattern:</b>

source cross-view/gibson4_exp.vars {Dataset_dir} {Bev_dir} {Log_Dir} {Train_split_path} {Val_split_path}


<b>E.g.:</b>

source cross-view/gibson4_exp.vars /home/shantanu/gibson4_dataset /home/shantanu/gibson4_dataset/dilated_partialmaps /home/shantanu/indolayout_logs /home/shantanu/indoor-layout-estimation-main/splits/gibson4/filtered_front_train_files.txt /home/shantanu/indoor-layout-estimation-main/splits/gibson4/filtered_front_val_files.txt

### Training options
Please refer to the cross-view/opt.py file for training/evaluation options available.


### Training Indolayout model
The main script for training the indolayout model is:

python3 train_disc.py --model_name attention_transformer_discr --data_path /home/shantanu/gibson4_dataset --split gibson4 --width 512 --height 512 --num_class 3 --type static --static_weight 1 --occ_map_size 128 --log_frequency 1 --log_root /home/shantanu/basic_discr --save_path /home/shantanu/basic_discr --semantics_dir None --chandrakar_input_dir None --floor_path None --batch_size 8 --num_epochs 100 --lr_steps 50 --lr 1e-4 --lr_transform 1e-3 --load_weights_folder None --bev_dir /scratch/shantanu/gibson4_dataset/dilated_partialmaps --train_workers 15 --val_workers 8


### Evaluate Indolayout
To evaluate the same model, simply change the script to 'eval.py' as follows:

eval.py --model_name attention_transformer_discr --data_path /home/shantanu/gibson4_dataset --split gibson4 --width 512 --height 512 --num_class 3 --type static --static_weight 1 --occ_map_size 128 --log_frequency 1 --log_root /home/shantanu/basic_discr --load_weights_folder /home/shantanu/basic_discr/epoch_100 --semantics_dir None --chandrakar_input_dir None --floor_path None --batch_size 8 --num_epochs 1 --bev_dir /scratch/shantanu/gibson4_dataset/dilated_partialmaps --train_workers 0 --val_workers 8



## Supplementary Material

Refer to the notebooks folder to understand experiments and their implementations individually, in particular for:
1. data_visualization
2. generate_visible_occupancy
3. photometric_reconstruction
4. evaluate_bev




