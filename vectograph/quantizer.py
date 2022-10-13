from sklearn.base import BaseEstimator, TransformerMixin
from vectograph.utils import create_experiment_folder
from vectograph.transformers import KGSave, GraphGenerator
import sklearn
import pandas as pd


class QCUT(BaseEstimator, TransformerMixin):
    """
    Quantile-based discretization function based on Pandas(
    https://pandas.pydata.org/docs/reference/api/pandas.qcut.html)
    """

    def __init__(self, min_unique_val_per_column=1, num_quantile=4,
                 remove_old_numerical_values=True, path=None, duplicates='raise'):
        """

        :type storage_path: str
        """
        self.min_unique_values_per_feature = min_unique_val_per_column
        self.num_quantile = num_quantile
        self.remove_old_numerical_values = remove_old_numerical_values
        self.duplicates = duplicates

        if path is None:
            self.path, _ = create_experiment_folder()

    def fit(self, x, y=None):
        """
        :param x:
        :param y:
        :return:
        """
        return self

    @staticmethod
    def __sanity_checking(x):
        try:
            assert isinstance(x, pd.DataFrame)
        except AssertionError:
            print(f'Input {type(x)} is not a pandas.DataFrame...')
            x=pd.DataFrame(x)
            print(f'Input is converted to a pandas.DataFrame {type(x)}...')
        try:
            for i in x.columns:
                assert isinstance(i, str)
        except AssertionError:
            print('Column names are not string. => ', [(i, type(i)) for i in x.columns])
            print('Column names will be converted to string')
            x.columns = [str(i) for i in x.columns]
        return x

    def __perform_discretization(self, column_name: str, df: pd.DataFrame):
        """
        Given a vector of values that are stored in pandas series.
        Apply qcut on them with given paramster.

        discretized is a pandas Series that contrains quantiles (categories).

        bin_values: represents values.
        0    8.3252   => 0_quantile_3
        1    8.3014   => 0_quantile_3
        2    7.2574   => 0_quantile_3
        3    5.6431   => 0_quantile_3
        4    3.8462   => 0_quantile_2

        0    0_quantile_3
        1    0_quantile_3
        2    0_quantile_3
        3    0_quantile_3
        4    0_quantile_2

        Categories (4, object): [0_quantile_0 < 0_quantile_1 < 0_quantile_2 < 0_quantile_3]
        [ 0.4999   2.5634   3.5348   4.74325 15.0001 ]

        :param df_column:
        :param labels:
        :return:
        """

        # 3. Generate placeholders.
        labels = [column_name + '_quantile_' + str(i) for i in range(self.num_quantile)]
        # 4. Apply the Quantile-based discretization function
        try:
            discretized, bin_values = pd.qcut(x=df[column_name], q=self.num_quantile, retbins=True, labels=labels,
                                              duplicates=self.duplicates)
        except ValueError as e:
            print('#' * 10, end=' ')
            print(f'Error at applying Quantile-based discretization function (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html)')
            print(f'Number of quantiles per column/feature: {self.num_quantile} ')
            print(f'Number of unique values of the column/feature {column_name}: {len(df[column_name].unique())}')
            print(
                f'Either reduce the number of quantile parameter or set the duplicates parameter to ***drop*** (currently {self.duplicates})')
            raise e
        # 5. if column contains ***/*** => # substring: from the index of last / till the end.
        name_file = column_name[column_name.rfind('/') + 1:]
        # 6. Save the mappings from bins to values.
        pd.DataFrame.from_dict(dict(zip(discretized.cat.categories.tolist(), bin_values)), orient='index').to_csv(
            self.path + '/Feature_Category_' + name_file + '_Mapping.csv')
        return 'Feature_Category_' + column_name, discretized, bin_values

    def transform(self, df: pd.DataFrame):
        """
        Input data is discretized via QCUT function and stored as pandas dataframe
        :param df: df is expected to be pandas Dataframe
        :return df: A discretized tabular data
        """
        print('Original Tabular data: {0} by {1}'.format(*df.shape))
        df = self.__sanity_checking(df)

        columns_to_drop = []
        for col in df.select_dtypes(exclude='object').columns:
            # 1. Check whether number of unique values in this respective column is greater than input constraint.
            if len(df[col].unique()) >= self.min_unique_values_per_feature:
                # 2. Remember the column.
                columns_to_drop.append(col)
                new_column_name, discretized, bin_values = self.__perform_discretization(column_name=col, df=df)

                # 3. Create new column containing discretized values.
                df.loc[:, new_column_name] = discretized
        if self.remove_old_numerical_values:
            df.drop(columns=columns_to_drop, inplace=True)
        return df
