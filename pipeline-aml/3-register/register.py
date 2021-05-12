import os
import argparse
from azureml.core import Run

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)  
    parser.add_argument('--name_model_file', type=str)
    parser.add_argument('--name_model', type=str)      
    args = parser.parse_args()
    return args

def main():
    args = get_runtime_args()
    run = Run.get_context()
    
    full_path = os.path.join(args.model_path, args.name_model_file)
    # Ironico hacer esto, ya que el archivo existe en el datastore pero igual toca subirlo
    # https://stackoverflow.com/a/59694422
    run.upload_file(full_path, full_path)
    run.register_model(
        model_path=full_path,
        model_name=args.name_model
    )

if __name__ == "__main__":
    main()