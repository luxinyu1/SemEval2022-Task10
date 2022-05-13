CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "norec" \
                            --plm_model_name "nb-bert-base" \
                            --batch_size 32 \
                            --seed 1 \
                            --do_train True \