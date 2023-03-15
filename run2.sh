	for layer in 1
	do
    gpu_id=1
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 main_add_orthogonal_2gnn.py --config_file new_config.yml \
                --fine_tune \
                --add_adj \
                --adj_dir adj/bert-base-cased/bert-attn-mean-eye-norm/bert-attn-layer${layer} \
                --save_dir ckpt/ace2005_bert/bert-base-cased/bert-attn-mean-eye-norm/bert-attn-layer${layer}/5216/add_orthogonal_uni_2gnn \
                --device 0 \
                --gcn_layers 1 \
                --is_unidirectional
  done
#                --gcn_dropout 0.0 \
#                --adj_dropout 0.1 \
#                --gcn_layers 1 \
#                --device 0
#done
