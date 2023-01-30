	for layer in 9 10 11
	do
    gpu_id=3
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 main.py --config_file new_config.yml \
                --data_dir data/entity-relation/SciERC \
                --save_dir ckpt/scierc_scibert/scibert_scivocab_uncased/bert-attn-mean-eye-norm/bert-attn-layer${layer} \
                --bert_model_name /mnt/data1/public/pretrain/scibert_scivocab_uncased \
                --fine_tune \
                --epochs 300 \
                --early_stop 50 \
                --add_adj \
                --adj_dir adj/scibert_scivocab_uncased/bert-attn-mean-eye-norm/bert-attn-layer${layer} \
                --gcn_layers 1 \
                --device 0
  done
