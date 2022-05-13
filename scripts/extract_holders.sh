for dataset in "opener_en" "mpqa" "darmstadt_unis"; do

    echo "===========${dataset}==========="

    python ./src/GTS/extract_holders.py --dataset $dataset \
                                            --plm_model_name "ernie_2.0_skep_large_en_pytorch" \

done;

for dataset in "multibooked_ca" "multibooked_eu" "norec" "opener_es"; do

    echo "===========${dataset}==========="

    python ./src/GTS/extract_holders.py --dataset $dataset \
                                            --plm_model_name "xlm-roberta-large" \

done;