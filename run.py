
import numpy as np

import spotlight
from spotlight.validator import Validator
from spotlight.datasets.movielens import get_movielens_dataset


dataset = get_movielens_dataset(variant='100K')
print(dataset)