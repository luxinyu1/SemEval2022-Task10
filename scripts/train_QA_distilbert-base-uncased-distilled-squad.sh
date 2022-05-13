CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "opener_en" \
                            --plm_model_name "distilbert-base-uncased-distilled-squad" \
                            --do_train True \
                            --num_train_epochs 15 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "darmstadt_unis" \
                            --plm_model_name "distilbert-base-uncased-distilled-squad" \
                            --do_train True \
                            --num_train_epochs 15 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_QA.py --dataset "mpqa" \
                            --plm_model_name "distilbert-base-uncased-distilled-squad" \
                            --do_train True \
                            --num_train_epochs 15 \