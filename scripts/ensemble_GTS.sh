CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_GTS.py --dataset "opener_en" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "ernie_2.0_skep_large_en_pytorch" "bert-large-uncased"\
                                    # --ensemble_weights 0.5 0.5 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_GTS.py --dataset "mpqa" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "ernie_2.0_skep_large_en_pytorch" "bert-large-uncased"\
                                    # --ensemble_weights 0.5 0.5 \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_GTS.py --dataset "darmstadt_unis" \
                                    --batch_size 4 \
                                    --ensemble_plm_model_names "ernie_2.0_skep_large_en_pytorch" "bert-large-uncased"\
                                    # --ensemble_weights 0.5 0.5 \