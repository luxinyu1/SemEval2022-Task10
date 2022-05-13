for dataset in "opener_en" "mpqa" "darmstadt_unis"; do
    CUDA_VISIBLE_DEVICES=$1 python ./src/GTS/train_GTS.py --dataset $dataset \
                                --plm_model_name "bert-base-uncased" \
                                --seed 1 \
                                --batch_size 16 \
                                --nhops 3 \
                                --learning_rate 3e-5 \
                                --docker_mode True \
                                --disable_progress_bar True \
                                # --no_cuda
done