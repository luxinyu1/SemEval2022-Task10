for dataset in "opener_en" "mpqa" "darmstadt_unis"; do

    echo "Converting" "${dataset}" "..."
    python ./src/GTS/prepare_data_for_QA_task.py --prepare_for_step2\
                                        --step1_dataset "${dataset}" \
                                        --mode "eval";
done

for dataset in "multibooked_ca" "multibooked_eu" "norec" "opener_es"; do

    echo "Converting" "${dataset}" "..."

    python ./src/GTS/prepare_data_for_QA_task.py --prepare_for_step2\
                                        --step1_dataset "${dataset}" \
                                        --mode "eval";

done

for dataset in "opener_en" "mpqa" "darmstadt_unis"; do

    echo "Converting" "${dataset}" "..."
    python ./src/GTS/prepare_data_for_QA_task.py --prepare_for_step2\
                                        --step1_dataset "${dataset}" \
                                        --mode "predict";
done

for dataset in "multibooked_ca" "multibooked_eu" "norec" "opener_es"; do

    echo "Converting" "${dataset}" "..."

    python ./src/GTS/prepare_data_for_QA_task.py --prepare_for_step2\
                                        --step1_dataset "${dataset}" \
                                        --mode "predict";

done