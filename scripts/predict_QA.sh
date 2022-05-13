CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "opener_en" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "mpqa" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "darmstadt_unis" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "opener_es" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "norec" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "multibooked_ca" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "multibooked_eu" \
                            --do_predict True \
                            --per_device_eval_batch_size 16 \