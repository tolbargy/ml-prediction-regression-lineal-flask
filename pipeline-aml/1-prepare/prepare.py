import os
import argparse
import pandas as pd

def get_runtime_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-input-path', type=str)
    parser.add_argument('--data-output-path', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_runtime_args()

    # Crear un directorio de salida
    os.makedirs(args.data_output_path, exist_ok=True)

    input_file_path = os.path.join(args.data_input_path, 'compensation_dataset.csv')
    output_file_path = os.path.join(args.data_output_path, 'compensation_dataset.csv')

    print(f'Reading data from {input_file_path} and writing processed output to {output_file_path}')
    print(f'Output dir: {os.listdir(args.data_output_path)}')
    # Leemos el archivo de entrada .csv
    compensation_dataset = pd.read_csv(input_file_path)

    # Algún preprocesamiento de datos debería ocurrir aquí ...
    pass

    # Luego del preprocesamiento que hicimos a los datos, lo enviamos al archivo de salida
    compensation_dataset.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main()