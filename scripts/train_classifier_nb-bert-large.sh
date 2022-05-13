CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "norec" \
                            --plm_model_name "nb-bert-large" \
                            --batch_size 64 \
                            --seed 1 \
                            --do_train True \