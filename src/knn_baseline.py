import numpy as np
import pandas as pd
import spotlight
from spotlight.datasets.movielens import get_movielens_dataset
import sys
from spotlight.datasets.synthetic import generate_sequential
from surprise import KNNBasic
from surprise import BaselineOnly
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse


def knn_base():
    data = Dataset.load_builtin('ml-100k')
    trainset, testset = train_test_split(data, test_size=.5)
    algo = KNNBasic()
    # Train the algorithm on the trainset, and predict ratings for the testset
    algo.fit(trainset)
    predictions = algo.test(testset)
    # Then compute RMSE
    score = accuracy.rmse(predictions)
    return score
    
