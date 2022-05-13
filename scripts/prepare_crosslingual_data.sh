python ./src/GTS/prepare_crosslingual_data.py --use_dataset opener_en opener_es mpqa multibooked_eu \
                                                    --predict_set multibooked_ca

python ./src/GTS/prepare_crosslingual_data.py --use_dataset opener_en opener_es mpqa multibooked_ca \
                                                    --predict_set multibooked_eu

python ./src/GTS/prepare_crosslingual_data.py --use_dataset opener_en mpqa multibooked_ca multibooked_eu \
                                                    --predict_set opener_es