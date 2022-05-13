CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "opener_en" \
                            --plm_model_name "bert-large-uncased" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-5 \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "mpqa" \
                            --plm_model_name "bert-large-uncased" \
                            --seed 1 \
                            --batch_size 16 \
                            --nhops 2 \
                            --num_warmup_steps 2000 \
                            --learning_rate 3e-6 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "darmstadt_unis" \
                            --plm_model_name "bert-large-uncased" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-6 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda