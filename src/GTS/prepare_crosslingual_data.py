import os
import json
import shutil
import argparse
import logging
import pickle

from utils import DATA_DIR

logger = logging.getLogger(__name__)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--use_dataset',
                        nargs='+',
                        required=True)

    parser.add_argument('--predict_set',
                        required=True,
                        choices=["multibooked_ca", "multibooked_eu", "opener_es"],
                        )
                        
    args = parser.parse_args()

    # Sanity Check

    if (args.predict_set == "multibooked_ca" and "multibooked_ca" in args.use_dataset) \
        or (args.predict_set == "multibooked_eu" and "multibooked_eu" in args.use_dataset) \
        or (args.predict_set == "opener_es" and "opener_es" in args.use_dataset):

        raise ValueError("This is a zero-shot task!")

    return args

def main():

    args = parse_args()

    all_training_set = []
    # all_validation_set = []
    all_holder_dict = {}

    for ds_name in args.use_dataset:

        train_ds = json.load(open(DATA_DIR / ds_name / 'train.json'))
        # validation_ds = json.load(open(DATA_DIR / ds_name / 'dev.json'))

        all_training_set.extend(train_ds)
        # all_validation_set.extend(validation_ds)
        holder_dict_path = DATA_DIR / 'holder_dict' / ds_name / 'holder_dict.pickle'

        with open(holder_dict_path, 'rb') as f_pickle:
            holder_dict = pickle.load(f_pickle)
        
        all_holder_dict.update(holder_dict)

    os.makedirs(DATA_DIR / f"crosslingual_{args.predict_set}", exist_ok=True)
    os.makedirs(DATA_DIR / "holder_dict" / f"crosslingual_{args.predict_set}", exist_ok=True)

    with open(DATA_DIR / f"crosslingual_{args.predict_set}" / "train.json", 'w+') as f_train:

        json.dump(all_training_set, f_train)

    shutil.copyfile(DATA_DIR / args.predict_set / 'dev.json', DATA_DIR / f"crosslingual_{args.predict_set}" / 'dev.json')
    shutil.copyfile(DATA_DIR / args.predict_set / 'test.json', DATA_DIR / f"crosslingual_{args.predict_set}" / 'test.json')
    
    with open(DATA_DIR / "holder_dict" / f"crosslingual_{args.predict_set}" / "holder_dict.pickle", 'wb') as f_pickle:
        pickle.dump(all_holder_dict, f_pickle)

if __name__ == "__main__":
    main()