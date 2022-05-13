CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "crosslingual_multibooked_ca" \
                            --plm_model_name "xlm-roberta-large" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-6 \
                            --num_warmup_steps 1000 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            --save_last_epoch True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "crosslingual_multibooked_eu" \
                            --plm_model_name "xlm-roberta-large" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-6 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            --save_last_epoch True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "crosslingual_opener_es" \
                            --plm_model_name "xlm-roberta-large" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-6 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            --save_last_epoch True \
                            # --no_cuda