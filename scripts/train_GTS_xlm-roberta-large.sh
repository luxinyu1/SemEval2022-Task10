CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "multibooked_ca" \
                            --plm_model_name "xlm-roberta-large" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-5 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "multibooked_eu" \
                            --plm_model_name "xlm-roberta-large" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-5 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "norec" \
                            --plm_model_name "xlm-roberta-large" \
                            --seed 1 \
                            --batch_size 16 \
                            --nhops 3 \
                            --learning_rate 3e-5 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "opener_es" \
                            --plm_model_name "xlm-roberta-large" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-5 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda