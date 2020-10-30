# checkpoint 1

to execute the code please run these two commands: 

pip install -r requirments.txt 

python run.py

*the dataset is installed directly by a function in the spotlight.datasets.movielens module of the spotlight package, and so it does not need a data-params.json file for input 

The run.py file imports numpy, spotlight and sub-modules datasets and cross_validation, it then installs the dataset into the machine through the function get_movielens_dataset. It then uses the random_train_test_split function to split the data into training and test data with an 80:20 split.
