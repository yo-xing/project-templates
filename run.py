import sys
import numpy as np

import spotlight
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.datasets.synthetic import generate_sequential
from spotlight.cross_validation import random_train_test_split

sys.path.insert(0, 'src')
from data import generate_data, save_data
from main_model import build_model
from mean_baseline import build_mean_baseline_model

def main(targets):
    data_config = json.load(open('config/data-params.json'))
    main_model_config = json.load(open('config/main-model-params.json'))
    
    if 'test' in targets:
        dataset = generate_data(**data_config)
        save_data(dataset, **data_config)
        
        main_rsme = build_model(dataset, **main_model_config)
        mean_baseline_rsme = build_mean_baseline_model(dataset)


if __name__ == "__main__":
    main()