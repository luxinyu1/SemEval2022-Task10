CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_classifier.py --dataset "opener_en" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "bert-large-uncased" "ernie_2.0_skep_large_en_pytorch" \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_classifier.py --dataset "mpqa" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "bert-base-uncased" "bert-large-uncased" \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_classifier.py --dataset "darmstadt_unis" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "bert-base-uncased" "bert-large-uncased" \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_classifier.py --dataset "opener_es" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "bert-base-multilingual-cased" "LaBSE" \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_classifier.py --dataset "multibooked_ca" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "bert-base-multilingual-cased" "LaBSE" \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_classifier.py --dataset "multibooked_eu" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "bert-base-multilingual-cased" "LaBSE" \

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/ensemble_classifier.py --dataset "norec" \
                                    --batch_size 8 \
                                    --ensemble_plm_model_names "bert-base-multilingual-cased" "nb-bert-base" "nb-bert-large" "LaBSE" \