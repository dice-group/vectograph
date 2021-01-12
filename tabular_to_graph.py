from sklearn.base import BaseEstimator, TransformerMixin
from vectograph.utils import create_experiment_folder
from vectograph.transformers import KGCreator, SimpleKGCreator
from vectograph.quantizer import QCUT

import sklearn
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression

X, y = datasets.fetch_california_housing(return_X_y=True)
print(cross_val_score(LinearRegression(), X, y, cv=10).mean())

storage_path, _ = create_experiment_folder()
X_transformed = QCUT(path=storage_path).transform(pd.DataFrame(X))

# Add prefix
X_transformed.index = 'Event_' + X_transformed.index.astype(str)
kg = SimpleKGCreator().transform(X_transformed)

for s, p, o in kg:
    print(s, p, o)
