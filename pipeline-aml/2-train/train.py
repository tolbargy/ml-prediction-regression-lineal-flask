import numpy as np
import pandas as pd
import pickle
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-input-path', type=str)
    args = parser.parse_args()
    return args

def feature_engineer_data(dataset_path):
    dataset = pd.read_csv(dataset_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def main():
    args = get_runtime_args()
    print('ARGS Test:')
    print(args)
    print(args.data_input_path)
    dataset_path = os.path.join(args.data_input_path, 'compensation_dataset.csv')
    X_train, X_test, y_train, y_test = feature_engineer_data(dataset_path)

    model = train_model(X_train, y_train)

if __name__ == '__main__':
    main()
