# Vectograph

Vectograph is an open-source software library for automatically creating a graph structured data from a given tabular data.

- [Installation](#installation)

# Installation
### Installation from source
```
git clone https://github.com/dice-group/Vectograph.git
conda create -n temp python=3.6 # Or be sure that your have Python => 3.6.
conda activate temp
pip install -e . 
python -c "import vectograph"
python -m pytest tests
```
### Installation via pip (later)

```
pip install vectograph
```

### Using vectograph

```python
from vectograph.transformers import GraphGenerator
from vectograph.quantizer import QCUT
import pandas as pd
from sklearn import datasets

X, y = datasets.fetch_california_housing(return_X_y=True)
X_transformed = QCUT(min_unique_val_per_column=2, num_quantile=5).transform(pd.DataFrame(X))
# Add prefix
X_transformed.index = 'Event_' + X_transformed.index.astype(str)
kg = SimpleKGCreator().transform(X_transformed)
for s, p, o in kg:
    print(s, p, o)
```


## How to cite
If you want to cite the framework, feel free to
```
@article{demir2021vectograph,
  title={Vectograph},
  author={Demir, Caglar},
  journal={GitHub. Note: https://github.com/dice-group/Vectograph},
  volume={1},
  year={2021}
}
```

For any further questions, please contact:  ```caglar.demir@upb.de```