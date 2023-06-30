python3 train_posenet_bev.py configs/eval_bev.yaml \
model_name mb_sgbm_dpred_bpred \
data_path /scratch/shantanu.singh/gibson_data/HabitatGibson/data \
log_dir /scratch/shantanu.singh/preds \
load_weights_folder /scratch/shantanu.singh/indoor-layout-estimation/results/mb_sgbm_dpred_bpred/models/weights_49 \
depth_dir /scratch/shantanu.singh/gibson_data/HabitatGibson/sgbm_preddepth \
bev_dir /scratch/shantanu.singh/gibson_data/HabitatGibson/bevs/sim \
mode debug \
BEV_DECODER.n_channels 4