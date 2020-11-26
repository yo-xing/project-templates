
import numpy as np

import spotlight
from spotlight.datasets.movielens import get_movielens_dataset
import sys
from spotlight.datasets.synthetic import generate_sequential


print(sys.argv)

    
dataset = get_movielens_dataset(variant='100K')


if 'test' in sys.argv:
    dataset = generate_sequential()
    df2 = pd.DataFrame(dataset.tocoo().toarray())
    df2.to_csv('testdata_output.csv')
    
print('dataset: ', dataset)

from spotlight.cross_validation import random_train_test_split

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

print('Split into \n {} and \n {}.'.format(train, test))

