CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "crosslingual_opener_es" \
                            --plm_model_name "LaBSE" \
                            --num_warmup_steps 1000 \
                            --learning_rate 3e-6 \
                            --batch_size 128 \
                            --seed 1 \
                            --do_train True \
                            --crosslingual True \
                            --save_last_epoch True \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "crosslingual_multibooked_ca" \
                            --plm_model_name "LaBSE" \
                            --num_warmup_steps 1000 \
                            --learning_rate 3e-6 \
                            --batch_size 128 \
                            --seed 1 \
                            --do_train True \
                            --crosslingual True \
                            --save_last_epoch True \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "crosslingual_multibooked_eu" \
                            --plm_model_name "LaBSE" \
                            --num_warmup_steps 1000 \
                            --learning_rate 3e-6 \
                            --batch_size 128 \
                            --seed 1 \
                            --do_train True \
                            --crosslingual True \
                            --save_last_epoch True \