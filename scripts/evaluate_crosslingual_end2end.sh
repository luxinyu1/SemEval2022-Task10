for model in "LaBSE"; do

    for dataset in "crosslingual_opener_es" "crosslingual_multibooked_eu" "crosslingual_multibooked_ca"; do

        echo -e "$model" "$dataset \n"

        CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/run_classifier.py --dataset $dataset \
                                                --plm_model_name $model \
                                                --do_eval True \
                                                --crosslingual True ;
    
    done

done