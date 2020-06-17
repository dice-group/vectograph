import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from vectograph.utils import ignore_columns
from vectograph.helper_funcs import apply_PYKE
from vectograph.transformers import RDFGraphCreator

path_of_folder = '/home/demir/Desktop/data_files/'
tabular_csv_data_name = 'stock_change'  # (2903564, 15)
# DASK can be applied.
df = pd.read_csv(path_of_folder + tabular_csv_data_name + '.csv', index_col=0, low_memory=False)
# num_unique_values_per_column(df)
temp_df = ignore_columns(df.head(1000), ['customer_id', 'customer_name', 'customer_number',
                         'site_id', 'op_group_id', 'logistic_type',
                         'location_uuid', 'assortment_uuid', 'customer_item_number',
                         'supplier_id', 'supplier_item_number', 'box_number_in_site'])
temp_df.dropna(axis='columns', thresh=len(temp_df) // 3, inplace=True)  # drop columns having 30% NaN values.
# TODO Unfortunately, changed_at column could not recognized as datetime due to "T" in value ,i.e, 2015-04-16 T04:14:00
# TODO: takes quite some time.
temp_df['changed_at'] = pd.to_datetime(temp_df.changed_at)

temp_df.index = 'Event_' + temp_df.index.astype(str)

for col in temp_df.select_dtypes(include=['datetime', 'float']).columns:
    if len(temp_df[col].unique()) > 10:
        temp_df.loc[:, col + '_range'] = pd.qcut(temp_df[col], 10, labels=[col + '_quantile_' + str(i) for i in range(10)])

print(temp_df.head())
print(temp_df.shape)

kg_path = path_of_folder + tabular_csv_data_name
pipe = Pipeline([('createkg', RDFGraphCreator(path=kg_path, dformat='ntriples')),
                 ('embeddings', FunctionTransformer(apply_PYKE))])
pipe.fit_transform(X=temp_df.select_dtypes(include=['category', 'integer','float']))
