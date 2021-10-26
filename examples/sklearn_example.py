"""
====================================================================
From a tabular data to a knowledge graph
====================================================================
(1) Discretize variable/a column of the input tabular data into equal-sized buckets based on rank or based on sample quantiles.
(2) Construct a knowledge graph from (1)
    (2.1) Let X a discretized tabular data with m rows and n columns
    (2.2) We construct a knowledge graph from x by creating m  times n number of triples
    (2.3) Each column name is considered as a relation and each value in a cell corresponds to a node/entity

"""
from vectograph.transformers import GraphGenerator,KGSave
from vectograph.quantizer import QCUT
import pandas as pd
from sklearn import datasets

X, y = datasets.fetch_california_housing(return_X_y=True)
X_transformed = QCUT(min_unique_val_per_column=6, num_quantile=5).transform(pd.DataFrame(X))
# Add prefix
X_transformed.index = 'Event_' + X_transformed.index.astype(str)
kg = GraphGenerator(kg_path='.', kg_name='SimpleKG.nt').transform(X_transformed)
for s, p, o in kg:
    print(s, p, o)
