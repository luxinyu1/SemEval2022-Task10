for model in "bert-large-uncased" "bert-base-uncased" "ernie_2.0_skep_large_en_pytorch"; do
    for dataset in "opener_en" "mpqa" "darmstadt_unis"; do

        echo -e "$model" "$dataset \n"

        CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset $dataset \
                                                --plm_model_name $model \
                                                --do_eval True ;

    done
done

for model in "bert-base-multilingual-cased" "LaBSE"; do

    for dataset in "opener_es" "multibooked_eu" "multibooked_ca" "norec"; do

        echo -e "$model" "$dataset \n"

        CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset $dataset \
                                                --plm_model_name $model \
                                                --do_eval True ;
    
    done

done

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "opener_en" \
                                        --plm_model_name "LaBSE" \
                                        --do_eval True ;

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "norec" \
                                        --plm_model_name "nb-bert-base" \
                                        --do_eval True ;

CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset "norec" \
                                        --plm_model_name "nb-bert-large" \
                                        --do_eval True ;