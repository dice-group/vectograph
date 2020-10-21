import argparse
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from vectograph.utils import ignore_columns, create_experiment_folder, create_logger
from vectograph.helper_funcs import apply_PYKE
from vectograph.transformers import RDFGraphCreator, KGCreator, ApplyKGE, TypePrediction, ClusterPurity
import time

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tabularpath", type=str,
                        default='/home/demir/Desktop/ai4bd-smart-logistics/2020-06-26-ai4bd-smart-logistics/merged.csv',
                        nargs="?", help="Path of Tabular Data, i.e./.../data.csv")

    parser.add_argument("--base_uri", type=str, default='https://ai4bd.com/resource/', nargs="?", help="Base URI.")
    # Hyper parameters for conversion
    parser.add_argument("--num_of_quantiles", type=int, default=40, nargs="?",
                        help="q param in https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html")
    parser.add_argument("--min_num_of_unique_values_per_column", type=int, default=10, nargs="?",
                        help="Apply Quantile-based discretization function on those columns having at least such "
                             "unique values.")
    ##################################################################################################
    parser.add_argument("--model", type=str, default='Pyke', nargs="?",
                        help="Models:Distmult, Pyke")
    # hyperparameters of embedding models.
    parser.add_argument("--K_for_PYKE", type=int, default=10, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--embedding_dim", type=int, default=50, nargs="?",
                        help="Number of dimensions in embedding space.")
    parser.add_argument("--num_iterations", type=int, default=10, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?",
                        help="Batch size.")
    parser.add_argument("--input_dropout", type=float, default=0.1, nargs="?",
                        help="Dropout rate in input layer.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--evaluation", type=bool, default=False, nargs="?",
                        help="Quantify the quality of embeddings by applying an evaluation scenario depending on the KGE model.")

    args = parser.parse_args()
    tabular_path = args.tabularpath
    base_uri = args.base_uri
    num_of_quantiles = args.num_of_quantiles
    min_num_of_unique_values_per_col = args.min_num_of_unique_values_per_column
    eval_flag = args.evaluation
    # TODO 'Distmult' would not scale well on large KGE due to 1vsN.
    # We might want to use https://docs.ampligraph.org/en/1.3.2/
    possible_models = ['Pyke', 'Distmult']
    try:
        assert args.model in possible_models
    except:
        raise AssertionError('Given name for mode **{0}** is not found in {1}'.format(args.model, possible_models))
    params = {
        'model': args.model,
        'embedding_dim': args.embedding_dim,
        'num_iterations': args.num_iterations,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'input_dropout': args.input_dropout,
        'K_for_PYKE': args.K_for_PYKE}

    storage_path, _ = create_experiment_folder()
    logger = create_logger(name='Vectograph', p=storage_path)

    # DASK can be applied.
    df = pd.read_csv(tabular_path, low_memory=False)  # if dataset is very large use .head(1000)
    df.index = 'Event_' + df.index.astype(str)
    df.columns = [base_uri + i for i in df.columns]

    num_rows, num_cols = df.shape  # at max num_rows times num_cols columns.
    column_names = df.columns

    logger.info('Original Tabular data: {0} by {1}'.format(num_rows, num_cols))
    logger.info('Quantisation starts')
    for col in df.select_dtypes(exclude='object').columns:
        if len(df[col].unique()) >= min_num_of_unique_values_per_col:
            # labels indicates "the bins".
            labels = [col + '_quantile_' + str(i) for i in range(num_of_quantiles)]
            discretized, bin_values = pd.qcut(df[col].rank(method='first'),
                                              num_of_quantiles, retbins=True, labels=labels)
            df.loc[:, col + '_bin'] = discretized
            bins = discretized.cat.categories.tolist()
            name_file = col[col.rfind('/') + 1:]  # substring: from the index of last / till the end.
            pd.DataFrame.from_dict(dict(zip(bins, bin_values)), orient='index').to_csv(
                storage_path + '/Mapping_' + name_file + '.csv')

    new_num_rows, new_num_cols = df.shape  # at max num_rows times num_cols columns.

    logger.info('Tabular data after conversion: {0} by {1}'.format(new_num_rows, new_num_cols))

    params.update({'storage_path': storage_path,
                   'logger': logger})
    if eval_flag:
        pipe = Pipeline([('createkg', KGCreator(path=storage_path, logger=logger)),
                         ('embeddings', ApplyKGE(params=params)),
                         ('typeprediction', TypePrediction()),
                         ('clusterpruity', ClusterPurity())
                         ])
    else:
        pipe = Pipeline([('createkg', KGCreator(path=storage_path, logger=logger)),
                         ('embeddings', ApplyKGE(params=params))])

    df.to_csv(storage_path + '/ProcessedTabularData.csv')
    pipe.fit_transform(df)
