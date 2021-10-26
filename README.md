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
pip install vectograph # only a placeholder
```
### Using vectograph

```python
from vectograph.transformers import GraphGenerator
from vectograph.quantizer import QCUT
import pandas as pd
from sklearn import datasets

X, y = datasets.fetch_california_housing(return_X_y=True)
X_transformed = QCUT(min_unique_val_per_column=6, num_quantile=5).transform(pd.DataFrame(X))
# Add prefix
X_transformed.index = 'Event_' + X_transformed.index.astype(str)
kg = GraphGenerator().transform(X_transformed)

for s, p, o in kg:
    print(s, p, o)
```

### Scripting Vectograph & [DAIKIRI-Embedding](https://github.com/dice-group/DAIKIRI-Embedding)
From a tabular data to knowledge graph embeddings
```
# (1) Clone the repositories.
git clone https://github.com/dice-group/DAIKIRI-Embedding.git
git clone https://github.com/dice-group/vectograph.git
# (3) Create a virtual enviroment and install the dependicies pertaining to the DAIKIRI-Embedding framework.
conda env create -f DAIKIRI-Embedding/environment.yml
conda activate daikiri
# (4) Install dependencies of the vectograph framework.
cd vectograph
pip install -e .
cd ..
# (5) Create a knowledge graph by using an example dataset from sklearn.datasets.fetch_california_housing.html
python vectograph/main.py --kg_name "ExampleKG.nt"
# (6) Preperate data in requirement format for learning embeddings
mkdir DefaultKGExample
mv ExampleKG.nt DefaultKGExample/train.txt
# (7) Generate Embeddings
python DAIKIRI-Embedding/main.py --path_dataset_folder 'DefaultKGExample' --model 'ConEx'
# Result: A folder named with current time created that contains
# info.log, ConEx_entity_embeddings.csv, ConEx_relation_embeddings.csv, etc.
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