import numpy as np
import spotlight
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    dataset = get_movielens_dataset(variant='100K')
    return dataset

def to_df(x):
    df = pd.DataFrame(x.tocoo().toarray())
    df = df.replace(0.0, np.nan)
    return df


def plot_kde(dataset):
    s = pd.Series(dataset.ratings)
    ax = s.plot.kde(title = 'Kernal Density Estimate of User Ratings')
    ax.figure.savefig('kde_rating.png')

def plot_hist(dataset):
    s = pd.Series(dataset.ratings)
    ax = s.plot.hist(bins = 5, title = 'Histogram of User Ratings')
    ax.figure.savefig('hist_rating.png')
    
    
def plot_mean(df):
    mean_ratings = df.mean(skipna = True)
    ax = mean_ratings.plot.hist(title = 'Histogram of Average Movie Ratings')
    ax.figure.savefig('hist_mean.png')
    
def plot_count(df):
    count_ratings = df.count()
    ax = count_ratings.plot.hist(title = 'Histogram of total ratings by Movie')
    ax.figure.savefig('hist_count.png')
    
def scatter(df):
    mean_ratings = df.mean(skipna = True)
    count_ratings = df.count()
    mr = pd.concat([mean_ratings, count_ratings], axis =1)
    mr.columns = ['mean', 'count']
    ax = mr.plot.scatter(x='count',y='mean', c='DarkBlue', title = 'Mean ratings Vs Total Ratings by Movie')
    ax.figure.savefig('scatter.png')
    
def main():
    dataset = get_data()
    df = to_df(dataset)
    plot_kde(dataset)
    plot_hist(dataset)
    plot_mean(df)
    plot_count(df)
    scatter(df)
    
if __name__ == "__main__":
    main()
    
