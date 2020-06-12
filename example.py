import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from vectograph.helper_funcs import apply_PYKE
from vectograph.transformers import RDFGraphCreator

# settings
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# path_of_tabular_data='data/stock_change.csv' # (2903564, 15)
path_of_folder = 'data/'
tabular_data_name = 'order_df'

df = pd.read_csv(path_of_folder + tabular_data_name + '.csv', index_col=0, low_memory=False)
################################# DATA CLEANING ###################################################################
df.drop(columns=['customer_id', 'site_id', 'deliver_mon', 'deliver_tue', 'deliver_wed', 'deliver_thu', 'deliver_fri',
                 'customer_number', 'order_number', 'every_week', 'supplier_id'], inplace=True)
df.dropna(axis='columns', thresh=len(df) // 3, inplace=True)
# Convert to data format. a
df['ordering_at'] = pd.to_datetime(df.ordering_at)
df['requesting_at'] = pd.to_datetime(df.requesting_at)
df['shipping_at'] = pd.to_datetime(df.shipping_at)
df['confirmed_at'] = pd.to_datetime(df.confirmed_at)
df['confirming_at'] = pd.to_datetime(df.confirming_at)
df['replenished_at'] = pd.to_datetime(df.replenished_at)
df.index = 'Event_' + df.index.astype(str)

for col in df.select_dtypes(include=['datetime', 'float']).columns:
    if len(df[col].unique()) > 10:
        df.loc[:, col + '_range'] = pd.qcut(df[col], 10, labels=[col + '_quantile_' + str(i) for i in range(10)])
################################# DATA CLEANING ###################################################################


kg_path = path_of_folder + tabular_data_name
pipe = Pipeline([('createkg', RDFGraphCreator(path=kg_path, dformat='ntriples')),
                 ('embeddings', FunctionTransformer(apply_PYKE))])
pipe.fit_transform(X=df.select_dtypes(include=['category', 'integer']))
