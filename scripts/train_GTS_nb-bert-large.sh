CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "norec" \
                            --plm_model_name "nb-bert-large" \
                            --seed 1 \
                            --batch_size 16 \
                            --nhops 3 \
                            --learning_rate 3e-5 \
                            --lr_scheduler linear \
                            --docker_mode True \
                            --disable_progress_bar True \