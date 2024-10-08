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
conda create -n vectograph python=3.6 # Or be sure that your have Python => 3.6.
conda activate vectograph
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

### Integration
From tabular data to knowledge graph embeddings : Scripting Vectograph & [Knowledge Graph Embeddings at Scale](https://github.com/dice-group/DAIKIRI-Embedding)
```bash
# (1) Clone the repositories.
git clone https://github.com/dice-group/dice-embeddings
git clone https://github.com/dice-group/vectograph.git
# (3) Create a virtual environment and install the dependencies pertaining frameworks.
conda create -n dice python=3.9.12 && conda activate dice
pip3 install -r dice-embeddings/requirements.txt
pip3 install -e vectograph/.
# (5) Create a knowledge graph by using an example dataset from sklearn.datasets wine or fetch_california_housing
python vectograph/create_toy_data.py --toy_dataset_name "boston"
python vectograph/main.py --tabularpath "boston.csv" --kg_name "boston.nt" --num_quantile=10 --min_unique_val_per_column=12
# (6) Preparation for KGE
# (6.1) Create an experiment folder and Move the RDF knowledge graph into (6.1) and rename it
mkdir Example && mv boston.nt Example/train.txt
# (7) Generate Embeddings
python dice-embeddings/main.py --path_dataset_folder "Example" --model "ConEx"
# A folder named with current time created that contains following files
# ConEx_entity_embeddings.npz    entity_to_idx.gzip  relation_to_idx.gzip
# ConEx_relation_embeddings.csv  idx_train_df.gzip   report.json
# configuration.json             model.pt            train_df.gzip
```

## How to cite


For any further questions, please contact:  ```caglar.demir@upb.de```
