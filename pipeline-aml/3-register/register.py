import os
import argparse
from azureml.core import Run

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solo_prueba', type=str)
    parser.add_argument('--otra_prueba_bien', type=str)
    #parser.add_argument('--path_test_propio', type=str)    
    args = parser.parse_args()
    return args

def main():
    args = get_runtime_args()
    context = Run.get_context()
    
    context.register_model(model_path='outputs/model.pkl',
                            model_name=model_name,

if __name__ == "__main__":
    main()