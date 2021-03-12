import argparse
import pandas as pd
from vectograph.quantizer import QCUT
from vectograph.transformers import GraphGenerator
import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tabularpath", type=str, default='',
                        nargs="?", help="Path of Tabular Data, i.e./.../data.csv")
    # Hyper parameters for conversion
    parser.add_argument("--num_quantile", type=int, default=2, nargs="?",
                        help="q param in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html")
    parser.add_argument("--min_unique_val_per_column", type=int, default=2, nargs="?",
                        help="Apply Quantile-based discretization function on those columns having at least such "
                             "unique values.")
    args = parser.parse_args()
    # DASK can be applied.
    print('Tabular data is being read')
    try:
        df = pd.read_csv(args.tabularpath)
    except FileNotFoundError:
        print('File not found, We will use california housing dataset from sklearn')
        from sklearn import datasets

        X, y = datasets.fetch_california_housing(return_X_y=True)
        df = pd.DataFrame(X)

    print('Original Tabular data: {0} by {1}'.format(*df.shape))
    print('Quantisation starts')
    X_transformed = QCUT(min_unique_val_per_column=args.min_unique_val_per_column,
                         num_quantile=args.num_quantile).transform(df)
    X_transformed.index = 'Event_' + X_transformed.index.astype(str)
    print('Graph data being generated')
    kg = GraphGenerator().transform(X_transformed)