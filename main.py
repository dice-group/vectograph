import argparse
import pandas as pd
from sklearn import datasets
from vectograph.quantizer import QCUT
from vectograph.transformers import GraphGenerator
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tabularpath", type=str, default=None,
                        nargs="?", help="Path of Tabular Data, i.e./.../data.csv")
    # Hyperparameters for conversion
    parser.add_argument("--num_quantile", type=int, default=2, nargs="?",
                        help="q param in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html")
    parser.add_argument("--min_unique_val_per_column", type=int, default=2, nargs="?",
                        help="Apply Quantile-based discretization function on those columns having at least such "
                             "unique values.")
    parser.add_argument("--kg_path", type=str, default='.', nargs="?",
                        help="Path for knowledge graph to be saved")
    parser.add_argument("--kg_name", type=str, default='DefaultKG.nt', nargs="?",
                        help="The name of a Knowledge graph in the ntriple format.")
    parser.add_argument("--duplicates", type=str, default='raise')

    args = parser.parse_args()
    if args.tabularpath is not None:
        try:
            tabular_data = pd.read_csv(args.tabularpath,index_col=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not read csv file in {args.tabularpath}")
    else:
        print(f'An input parameter (args.tabularpath is {args.tabularpath})...')
        print('Sklearn fetch_california_housing dataset is used...')
        tabular_data, _ = datasets.fetch_california_housing(return_X_y=True)

    print('Quantisation starts')
    X_transformed = QCUT(min_unique_val_per_column=args.min_unique_val_per_column,
                         num_quantile=args.num_quantile,duplicates=args.duplicates).transform(tabular_data)
    X_transformed.index = 'Event_' + X_transformed.index.astype(str)
    print('Graph data being generated')
    kg = GraphGenerator(kg_path=args.kg_path, kg_name=args.kg_name).transform(X_transformed)
    print('Done!')
