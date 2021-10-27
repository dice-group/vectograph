"""
====================================================================
Sample dataset creator
====================================================================
(1) Select a benchmark tabular dataset from sklearn
(2) Store (1) as dataframe
"""

import argparse
from sklearn.datasets import load_iris, fetch_california_housing, load_diabetes, load_digits,load_wine,load_breast_cancer
import pandas as pd
import os
fixed_dataset_names = ['boston', 'iris', 'diabetes', 'digits', 'wine', 'breast_cancer']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy_dataset_name", type=str, default='boston', help=f"Possible dataset {fixed_dataset_names}")
    parser.add_argument("--path_to_save", type=str, default=None, help="Please insert the absolute path with filename,e.g. /home/.../example.csv")
    args = parser.parse_args()
    if not (args.toy_dataset_name in fixed_dataset_names):
        raise ValueError(
            f'{dataset_name} is not a toy dataset provided within sklearn\tPossible datasets{fixed_dataset_names}')
    else:
        if args.toy_dataset_name == 'boston':
            X, y = fetch_california_housing(return_X_y=True)
        elif args.toy_dataset_name == 'iris':
            X, y = load_iris(return_X_y=True)
        elif args.toy_dataset_name == 'diabetes':
            X, y = load_diabetes(return_X_y=True)
        elif args.toy_dataset_name == 'digits':
            X, y = load_digits(return_X_y=True)
        elif args.toy_dataset_name == 'wine':
            X, y = load_wine(return_X_y=True)
        elif args.toy_dataset_name == 'breast_cancer':
            X, y = load_breast_cancer(return_X_y=True)
        else:
            raise ValueError(f'{args.toy_dataset_name} is not found in available datasets')
    print(f'Chosen dataset:{args.toy_dataset_name}')
    df = pd.DataFrame(X)
    df['labels'] = y
    if args.path_to_save is None:
        print(os.getcwd())
        df.to_csv(f'{os.getcwd()}/{args.toy_dataset_name}.csv')
    else:
        df.to_csv(f'{args.path_to_save}.csv')
