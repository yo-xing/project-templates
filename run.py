import sys
import numpy as np

import spotlight
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.datasets.synthetic import generate_sequential
from spotlight.cross_validation import random_train_test_split
import json
sys.path.insert(0, 'src')
from main_model import build_model
from mean_baseline import build_mean_baseline_model
from data import generate_data, save_data
from knn_baseline import knn_base
from first_baseline import first_base


def main(targets):
    data_config = json.load(open('config/data-params.json'))
    main_model_config = json.load(open('config/main-model-params.json'))
    
    if 'test' in targets:
        dataset = generate_data(**data_config)
        save_data(dataset, **data_config)
        first_baseline_rsme = first_base()
        knn_baseline_rsme = knn_base()
        main_rsme = build_model(dataset, **main_model_config)
        print('Main RMSE: ',main_rsme,'First baseline RMSE: ', first_baseline_rsme, 
              'KNN baseline RMSE: ', knn_baseline_rsme)

if __name__ == "__main__":
    main(sys.argv)