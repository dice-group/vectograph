# Vectograph

Vectograph is an open-source software library for applying knowledge graph embedding approaches on tabular data. 
To this end, Vectograph enables users to converts tabular data into RDF knowledge graph and apply KGE approaches.

- [Framework](#Framework)
        
- [Installation](#installation)

# Installation
### Installation from source
```
1) git clone https://github.com/dice-group/Vectograph.git
4) conda create -n temp python=3.6 # Or be sure that your have Python => 3.6.
5) conda activate temp
6) pip install -e . 
7) python -c "import vectograph"
```
### Installation via pip (later)

```python
pip install vectograph
```

## Workflow
```python
tabular_path = '...'
base_uri = '...'
min_num_of_unique_values_per_col = '...' # parameter for Quantile-based discretization.
num_of_quantiles = '...' # parameter for Quantile-based discretization.
params = { 'model': 'Pyke', 'embedding_dim': 50, 'num_iterations': 10, 'K_for_PYKE': 32}

df = pd.read_csv(tabular_path) # Note that dask.dataframe.read_parquet can be applied on large tabular data.
########################################################################################
############### Data Preprocessing through Quantile-based discretization ###############
# 1. Apply Quantile-based discretization for each column of the input tabular data.
# 2. We refer pandas.qcut() https://bit.ly/3kmJDeU.
########################################################################################
df.to_csv(storage_path + '/ProcessedTabularData.csv') # Store discretized tabular data.
########################################################################################
############### Knowledge Graph Embeddings through sklearn pipeline ###############
# 1. Create knowledge graph from discretized tabular data.
     # 1.1 We refer KGEOnRelationalData.ipynb for details.
# 2. Apply PYKE on (1).
# 3. Evaluate embedding in the type prediction and cluster purity tasks. https://arxiv.org/abs/2001.07418
pipe = Pipeline([('createkg', KGCreator(path=storage_path, logger=logger)),
                ('embeddings', ApplyKGE(params=params)),
                ('typeprediction', TypePrediction()),
                ('clusterpruity', ClusterPurity())])
pipe.fit_transform(df)
```
