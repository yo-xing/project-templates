import numpy as np
import spotlight
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_data():
    dataset = get_movielens_dataset(variant='100K')
    return dataset

def to_df(x):
    df = pd.DataFrame(x.tocoo().toarray())
    df = df.replace(0.0, np.nan)
    return df


def plot_kde(dataset, outdir):
    s = pd.Series(dataset.ratings)
    ax = s.plot.kde(title = 'Kernal Density Estimate of User Ratings')
    ax.figure.savefig(os.path.join(outdir, 'kde_rating.png'))

def plot_hist(dataset, outdir):
    s = pd.Series(dataset.ratings)
    ax = s.plot.hist(bins = 5, title = 'Histogram of User Ratings')
    ax.figure.savefig(os.path.join(outdir, 'hist_rating.png'))
    
    
def plot_mean(df, outdir):
    mean_ratings = df.mean(skipna = True)
    ax = mean_ratings.plot.hist(title = 'Histogram of Average Movie Ratings')
    ax.figure.savefig(os.path.join(outdir, 'hist_mean.png'))
    
def plot_count(df, outdir):
    count_ratings = df.count()
    ax = count_ratings.plot.hist(title = 'Histogram of total ratings by Movie')
    ax.figure.savefig(os.path.join(outdir, 'hist_count.png'))
    
def scatter(df, outdir):
    mean_ratings = df.mean(skipna = True)
    count_ratings = df.count()
    mr = pd.concat([mean_ratings, count_ratings], axis =1)
    mr.columns = ['mean', 'count']
    ax = mr.plot.scatter(x='count',y='mean', c='DarkBlue', title = 'Mean ratings Vs Total Ratings by Movie')
    ax.figure.savefig(os.path.join(outdir, 'scatter.png'))
    
def generate_stats(data, outdir, **kwargs):
    os.makedirs(outdir, exist_ok=True)
    df = to_df(data)
    plot_kde(data, outdir)
    plot_hist(data, outdir)
    plot_mean(df, outdir)
    plot_count(df, outdir)
    scatter(df, outdir)
    
if __name__ == "__main__":
    main()
    
