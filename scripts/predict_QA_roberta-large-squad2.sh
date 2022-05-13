CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "opener_en" \
                            --plm_model_name "roberta-large-squad2" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "mpqa" \
                            --plm_model_name "roberta-large-squad2" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "darmstadt_unis" \
                            --plm_model_name "roberta-large-squad2" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \