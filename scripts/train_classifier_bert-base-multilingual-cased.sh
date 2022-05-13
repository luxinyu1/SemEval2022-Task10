CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "opener_es" \
                            --plm_model_name "bert-base-multilingual-cased" \
                            --batch_size 32 \
                            --seed 1 \
                            --do_train True \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "norec" \
                            --plm_model_name "bert-base-multilingual-cased" \
                            --batch_size 32 \
                            --seed 1 \
                            --do_train True \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "multibooked_eu" \
                            --plm_model_name "bert-base-multilingual-cased" \
                            --batch_size 32 \
                            --seed 1 \
                            --do_train True \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "multibooked_ca" \
                            --plm_model_name "bert-base-multilingual-cased" \
                            --batch_size 32 \
                            --seed 1 \
                            --do_train True \