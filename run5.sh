	for layer in 0 4 9
	do
    gpu_id=4
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 main_2gnn.py --config_file new_config.yml \
                --fine_tune \
                --add_adj \
                --adj_dir adj/bert-base-cased/bert-attn-mean-eye-norm/bert-attn-layer${layer} \
                --save_dir ckpt/ace2005_bert/bert-base-cased/bert-attn-mean-eye-norm/bert-attn-layer${layer}/1_y_1_2gnn_sp15 \
                --gcn_layers 1 \
                --device 0 \
                --separate_threshold 1.5
  done
