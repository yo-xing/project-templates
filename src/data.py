import os
import numpy as np

from spotlight.datasets.movielens import get_movielens_dataset

def generate_data(size_variant, **kwargs):
    dataset = get_movielens_dataset(variant=size_variant)
    return dataset

def save_data(data, data_fp, **kwargs):
    os.makedirs(os.path.split(data_fp)[0], exist_ok=True)
    data.to_csv(data_fp, index=False)
    return 
