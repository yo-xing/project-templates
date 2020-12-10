import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn
import math

def train(train_data):
    avg_ratings = train_data.groupby('movieId').rating.mean().to_dict()
    for k, v in avg_ratings.items():
            avg_ratings[k] = round(v*2)/2
    mean_avg = round(train_data['rating'].mean()*2)/2
    return avg_ratings, mean_avg

def make_predictions(movieId, avg_ratings, mean_avg):
    if movieId in avg_ratings:
        return avg_ratings[movieId]
    else: 
        return mean_avg
    
def predict(test_data, avg_ratings, mean_avg):
    test_data['predictions'] = test_data['movieId'].apply(make_predictions, args=[avg_ratings, mean_avg])
    rmse = math.sqrt(sklearn.metrics.mean_squared_error(test_data['rating'], test_data['predictions']))
    return rmse

def build_mean_baseline_model(data):
    train_data, test_data = random_train_test_split(data)
    avg_ratings, mean_avg = train(train_data)
    rmse = predict(test_data, avg_ratings, mean_avg)
    return rmse