# Vectograph

Vectograph is an open-source software library for applying knowledge graph embedding approaches on tabular data. 
To this end, Vectograph enables users to converts tabular data into RDF knowledge graph and apply KGE approaches.

- [Framework](#Framework)
        
- [Installation](#installation)

# Installation
### Installation from source
```
1) git clone https://github.com/dice-group/Vectograph.git
2) conda create -n temp python=3.6.2 # Or be sure that your have Python => 3.6.
3) conda activate temp
4) python vectograph/setup.py install
# After you receive this Finished processing dependencies for vectograph==0.0.1
5) python -c "import vectograph"
6) git clone https://github.com/dice-group/PYKE.git into Vectograph
```
### Installation via pip (later)

```python
pip install vectograph
```

## Usage


```python
import pandas as pd
from sklearn.pipeline import Pipeline
from vectograph.transformers import ApplyKGE, KGCreator

path_of_folder = '/.../data_files/'
tabular_csv_data_name = 'example'  
storage_path='/../Folder'
df = pd.read_csv(path_of_folder + tabular_csv_data_name + '.csv', index_col=0, low_memory=False)
####################################
#### Data Preprocessing ####
####################################

pipe = Pipeline([('createkg', KGCreator(path=storage_path, logger=None)), # inclide logger object if necesseary
                 ('embeddings', ApplyKGE(params={'model': 'Distmult',  # Pyke
                                                 'embedding_dim': 10,
                                                 'batch_size': 256,
                                                 'num_iterations': 100,
                                                 'learning_rate':0.05,
                                                 'input_dropout': 0.1}),
                ('typeprediction', TypePrediction()),
                 ('clusterpruity', ClusterPurity()))])

pipe.fit_transform(X=df)
```
