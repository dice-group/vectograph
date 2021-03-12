from vectograph.transformers import GraphGenerator
from vectograph.quantizer import QCUT
import pandas as pd
from sklearn import datasets


class TestDefault:
    def test_default_QCUT(self):
        X, y = datasets.fetch_california_housing(return_X_y=True)
        n, m = X.shape
        X_transformed = QCUT(min_unique_val_per_column=2, num_quantile=5).transform(pd.DataFrame(X))
        # Add prefix
        X_transformed.index = 'Event_' + X_transformed.index.astype(str)

        gg = GraphGenerator()
        kg = gg.transform(X_transformed)
        assert len(kg) == (n * m)
        new_kg = []
        with open(gg.path, 'r') as read:
            for i in read:
                s, p, o, = i.split()
                new_kg.append((s, p, o))

        assert kg == new_kg
