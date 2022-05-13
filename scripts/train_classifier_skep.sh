# CUDA_VISIBLE_DEVICES=$1 accelerate launch ./src/GTS/run_classifier.py --dataset "opener_en" \
#                             --plm_model_name "ernie_2.0_skep_large_en_pytorch" \
#                             --batch_size 64 \
#                             --learning_rate 3e-5 \
#                             --seed 1 \
#                             --do_train True \

# CUDA_VISIBLE_DEVICES=$1 accelerate launch ./src/GTS/run_classifier.py --dataset "mpqa" \
#                             --plm_model_name "ernie_2.0_skep_large_en_pytorch" \
#                             --num_warmup_steps 1000 \
#                             --batch_size 64 \
#                             --learning_rate 3e-6 \
#                             --seed 1 \
#                             --do_train True \
#                             --num_train_epochs 5 \

CUDA_VISIBLE_DEVICES=$1 accelerate launch ./src/GTS/run_classifier.py --dataset "darmstadt_unis" \
                            --plm_model_name "ernie_2.0_skep_large_en_pytorch" \
                            --batch_size 64 \
                            --learning_rate 3e-5 \
                            --seed 1 \
                            --do_train True \