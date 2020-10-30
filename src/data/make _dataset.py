import numpy as np

from spotlight.datasets.movielens import get_movielens_dataset

dataset = get_movielens_dataset(variant='100K')
print(dataset)