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
    context = Run.get_context()
    
    context.register_model(
        model_path=os.path.join(args.model_path, args.name_model_file),
        model_name=args.name_model
    )

if __name__ == "__main__":
    main()