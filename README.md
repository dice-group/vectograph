# Vectograph

Vectograph is an open-source software library for automatically creating a graph structured data from a given tabular data.

- [Creating Structured Data from Tabular Data](#creating-structured-data-from-tabular-data)
- [Installation](#installation)
- [Examples](#examples)

## Creating Structured Data from Tabular Data
Let **X** be a **m** by **n** matrix representing the input tabular, the structured data is created by following these steps:
1. Apply [QCUT algorithm](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html) for each column that has at least **min_unique_val_per_column** number of unique values.
2. Consider 
   1. **the i.th row** as the i.th [concise bounded description](https://www.w3.org/Submission/CBD/) of **the i.th event**.
   2. **the j.th column** as the j.th relation/predicate/edge.
   3. A triple is modeled as event_i -> relation_j -> **X_ij**.

Assume that we have the first row of [fetch_california_housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) is 
```
[   8.3252       41.            6.98412698    1.02380952  322. 2.55555556   37.88       -122.23      ]
```
Applying the QCUT algorithm with default parameters **min_unique_val_per_column=6, num_quantile=5** generates 0.th CBD
```    
<Event_0> <Feature_Category_0> <0_quantile_4> .
<Event_0> <Feature_Category_1> <1_quantile_4> .
<Event_0> <Feature_Category_2> <2_quantile_4> .
<Event_0> <Feature_Category_3> <3_quantile_1> .
<Event_0> <Feature_Category_4> <4_quantile_0> .
<Event_0> <Feature_Category_5> <5_quantile_1> .
<Event_0> <Feature_Category_6> <6_quantile_4> .
<Event_0> <Feature_Category_7> <7_quantile_0> .
```
that consist of **n** triples.
```<Feature_Category_0>``` represents the 0.th relation, i.e., 0.th column, whereas ```<0_quantile_4>``` represents a tail entity
, i.e., the 4.th bin of the 0.th column of the tabular data. . After the data conversion, we store each bin values. For instance, running examples/sklearn_example.py generates  ```Feature_Category_0_Mapping.csv``` that indicates
```0_quantile_4``` corresponds a bin that cover all values greater or equal than **5.10972**.

## Installation
```
git clone https://github.com/dice-group/Vectograph.git
conda create -n temp python=3.6 # Or be sure that your have Python => 3.6.
conda activate temp
pip install -e . 
python -c "import vectograph"
python -m pytest tests
```

## Examples
#### API Example
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

### Scripting Example
Create a toy dataset via sklearn. Available datasets: boston, iris, diabetes, digits, wine, and breast_cancer.
```bash
python create_toy_data.py --toy_dataset_name "boston"
# Discretize each column having at least 12 unique values into 10 quantiles, otherwise do nothing
python main.py --tabularpath "boston.csv" --kg_name "boston.nt" --num_quantile=10 --min_unique_val_per_column=12
```

### Scripting Vectograph & [DAIKIRI-Embedding](https://github.com/dice-group/DAIKIRI-Embedding)
From a tabular data to knowledge graph embeddings
```bash
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
python create_toy_data.py --toy_dataset_name "wine"
python main.py --tabularpath "wine.csv" --kg_name "wine.nt" --num_quantile=10 --min_unique_val_per_column=12
# (6) Preparation for DAIKIRI-Embedding
# (6.1) Create an experiment folder
mkdir DefaultKGExample
# (6.2) Move the RDF knowledge graph into (6.1) and rename it
mv wine.nt DefaultKGExample/train.txt
# (7) Generate Embeddings
python DAIKIRI-Embedding/main.py --path_dataset_folder 'DefaultKGExample' --model 'ConEx'
# Result: A folder named with current time created that contains
# info.log, ConEx_entity_embeddings.csv, ConEx_relation_embeddings.csv, etc.
```

## How to cite
If you really like this framework and want to cite it in your work, feel free to
```
@inproceedings{demir2021convolutional,
title={Convolutional Complex Knowledge Graph Embeddings},
author={Caglar Demir and Axel-Cyrille Ngonga Ngomo},
booktitle={Eighteenth Extended Semantic Web Conference - Research Track},
year={2021},
url={https://openreview.net/forum?id=6T45-4TFqaX}}
```

For any further questions, please contact:  ```caglar.demir@upb.de```