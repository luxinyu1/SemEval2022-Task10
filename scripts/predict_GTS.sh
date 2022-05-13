CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "opener_en" \
                            --batch_size 1 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "mpqa" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "darmstadt_unis" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "multibooked_ca" \
                            --batch_size 1 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "multibooked_eu" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "norec" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "opener_es" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "crosslingual_opener_es" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "crosslingual_multibooked_ca" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset "crosslingual_multibooked_eu" \
                            --batch_size 2 \
                            --docker_mode True \
                            --mode predict