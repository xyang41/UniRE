gpu_id=0
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 main.py --config_file new_config.yml \
                --test \
                --add_adj \
                --adj_dir adj/bert-base-cased/bert-attn-mean-eye-norm/bert-attn-layer11 \
                --save_dir ckpt/ace2005_bert/bert-base-cased/bert-attn-mean-eye-norm/bert-attn-layer11/1_y_1 \
                --gcn_layers 1 \
                --device 0 \
                --log_file test.log
  
