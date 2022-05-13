CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "multibooked_ca" \
                            --plm_model_name "xlm-roberta-large-squad2" \
                            --do_train True \
                            --per_device_train_batch_size 32 \
                            --per_device_eval_batch_size 32 \
                            --num_train_epochs 15 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "opener_es" \
                            --plm_model_name "xlm-roberta-large-squad2" \
                            --per_device_train_batch_size 32 \
                            --per_device_eval_batch_size 32 \
                            --num_warmup_steps 100 \
                            --do_train True \
                            --num_train_epochs 15 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "multibooked_eu" \
                            --plm_model_name "xlm-roberta-large-squad2" \
                            --per_device_train_batch_size 32 \
                            --per_device_eval_batch_size 32 \
                            --do_train True \
                            --num_train_epochs 20 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "norec" \
                            --plm_model_name "xlm-roberta-large-squad2" \
                            --per_device_train_batch_size 32 \
                            --per_device_eval_batch_size 32 \
                            --do_train True \
                            --num_train_epochs 15 \