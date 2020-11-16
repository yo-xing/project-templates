# checkpoint 2

to execute the code please run these two commands: 

pip install -r requirments.txt 

python run.py data eda

The dataset is installed directly by a function in the spotlight.datasets.movielens module of the spotlight package

The run.py file imports numpy, spotlight and sub-modules datasets and cross_validation, it then installs the dataset 
into the machine through the function get_movielens_dataset. It then uses the random_train_test_split function to 
split the data into training and test data with an 80:20 split. A new target was added called eda, which plots various graphs
to help visualize the data we are working with in this project. These graphs include a Kernal Density Estimate of user ratings
and a scatterplot of mean ratings vs total ratings by movie.

