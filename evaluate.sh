debugpy-run -p :5678 tmp.py --maxdisp 192 --with_spn --datapath /shanta_tmp/habitat_data \
   --save_path results/tmp --pretrained checkpoints/anynet_habitat/checkpoint.tar \
   --val_split_file splits/habitat/habitat_val_depth.txt \
   --test_bsize 1 --evaluate --save_eval_output