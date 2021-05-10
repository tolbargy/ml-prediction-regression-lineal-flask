import numpy as np
import pandas as pd
import pickle
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from azureml.core import Run

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str)
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

def predict(regressor, X_test):
    y_pred = regressor.predict(X_test)
    return y_pred

def export_model(regressor, model_path):
    pickle.dump(regressor, open(model_path,'wb'))

def import_model_predict(model_path, value, debug=False):
    """
        pickle.load() method loads the method and saves the deserialized bytes to model.
        Predictions can be done using model.predict().
        load again
    """
    model = pickle.load(open(model_path,'rb'))
    output = model.predict(value)
    if debug:
        print(output)
    return output

def execute_pipeline():
    args = get_runtime_args()
    dataset_path = os.path.join(args.data_path, 'compensation_dataset.csv')
    X_train, X_test, y_train, y_test = feature_engineer_data(dataset_path)

    model = train_model(X_train, y_train)

# run the code
if __name__ == '__main__':
    execute_pipeline()

