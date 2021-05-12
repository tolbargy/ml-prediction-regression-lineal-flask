import os
import argparse
from azureml.core import Run
import numpy as np
import pickle

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)    
    parser.add_argument('--name_model_file', type=str)
    args = parser.parse_args()
    return args

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

def main():
    args = get_runtime_args()
    run = Run.get_context()
    
    full_path = os.path.join(args.model_path, args.name_model_file)
    # Se valida la prediccion de salario para alguien con 5.3 años de experiencia
    data = 5.3
    value = [[np.array(data)]]
    salary_predict = import_model_predict(full_path, value=value, debug=True)
    run.log('Salary predict: ', salary_predict)

if __name__ == "__main__":
    main()