CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "opener_en" \
                            --plm_model_name "roberta-large-squad2" \
                            --do_train True \
                            --per_device_train_batch_size 16 \
                            --per_device_eval_batch_size 16 \
                            --num_train_epochs 15 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "mpqa" \
                            --plm_model_name "roberta-large-squad2" \
                            --do_train True \
                            --num_warmup_steps 1000 \
                            --per_device_train_batch_size 16 \
                            --per_device_eval_batch_size 16 \
                            --num_train_epochs 35 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "darmstadt_unis" \
                            --plm_model_name "roberta-large-squad2" \
                            --do_train True \
                            --num_warmup_steps 1000 \
                            --per_device_train_batch_size 16 \
                            --per_device_eval_batch_size 16 \
                            --num_train_epochs 20 \