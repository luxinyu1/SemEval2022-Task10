CUDA_VISIBLE_DEVICES=$1 accelerate launch ./src/GTS/run_classifier.py --dataset "opener_en" \
                            --plm_model_name "bert-large-uncased" \
                            --batch_size 16 \
                            --learning_rate 3e-6 \
                            --seed 1 \
                            --do_train True \

CUDA_VISIBLE_DEVICES=$1 accelerate launch ./src/GTS/run_classifier.py --dataset "mpqa" \
                            --plm_model_name "bert-large-uncased" \
                            --batch_size 16 \
                            --learning_rate 3e-6 \
                            --seed 1 \
                            --do_train True \

CUDA_VISIBLE_DEVICES=$1 accelerate launch ./src/GTS/run_classifier.py --dataset "darmstadt_unis" \
                            --plm_model_name "bert-large-uncased" \
                            --batch_size 16 \
                            --learning_rate 3e-6 \
                            --seed 1 \
                            --do_train True \