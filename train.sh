gpu_id=1
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 main_937.py --config_file config.yml \
                --fine_tune \
                --save_dir ckpt/ace2005_bert/base/937 \
                --device 0 \
                --seed 937
  
