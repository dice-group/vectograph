from vectograph.transformers import GraphGenerator
from vectograph.quantizer import QCUT
import pandas as pd
from sklearn import datasets


class TestDefault:
    def test_default_QCUT(self):

        X, y = datasets.fetch_california_housing(return_X_y=True)
        X_transformed = QCUT(min_unique_val_per_column=6, num_quantile=5).transform(pd.DataFrame(X))
        # Add prefix
        X_transformed.index = 'Event_' + X_transformed.index.astype(str)
        kg = GraphGenerator().transform(X_transformed)

        for s, p, o in kg:
            pass #print(s, p, o)
