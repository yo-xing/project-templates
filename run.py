
import numpy as np

import spotlight
from spotlight.datasets.movielens import get_movielens_dataset


dataset = get_movielens_dataset(variant='100K')
print('dataset: ', dataset)

from spotlight.cross_validation import random_train_test_split

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

print('Split into \n {} and \n {}.'.format(train, test))