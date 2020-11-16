import sys
import numpy as np

import spotlight
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split

sys.path.insert(0, 'src')
from data import generate_data, save_data
from eda import generate_stats 

def main(targets):
    data_config = json.load(open('config/data-params.json'))
    eda_config = json.load(open('config/eda-params.json'))

    if 'data' in targets:
        dataset = generate_data(**data_config)
        save_data(data, **data_config)

    if 'eda' in targets:
        try:
            dataset
        except NameError:
            dataset = pd.read_csv(data_config['data_fp'])
        generate_stats(dataset, **eda_config)


train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

print('Split into \n {} and \n {}.'.format(train, test))
