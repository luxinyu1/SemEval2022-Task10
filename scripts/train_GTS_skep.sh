CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "opener_en" \
                            --plm_model_name "ernie_2.0_skep_large_en_pytorch" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --learning_rate 3e-5 \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "mpqa" \
                            --plm_model_name "ernie_2.0_skep_large_en_pytorch" \
                            --seed 1 \
                            --batch_size 16 \
                            --nhops 2 \
                            --num_warmup_steps 1000 \
                            --learning_rate 3e-5 \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "darmstadt_unis" \
                            --plm_model_name "ernie_2.0_skep_large_en_pytorch" \
                            --seed 1 \
                            --batch_size 8 \
                            --nhops 3 \
                            --num_warmup_steps 500 \
                            --learning_rate 3e-5 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \
                            # --no_cuda