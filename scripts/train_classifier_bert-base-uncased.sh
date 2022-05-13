# CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "opener_en" \
#                             --seed 1 \
#                             --do_train True \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "mpqa" \
                            --batch_size 64 \
                            --seed 1 \
                            --do_train True \

# CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "darmstadt_unis" \
#                             --batch_size 32 \
#                             --seed 1 \
#                             --do_train True \