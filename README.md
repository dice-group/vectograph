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
4) python ontolearn/setup.py install
# After you receive this Finished processing dependencies for OntoPy==0.0.1
5) python -c "import vectograph"
```
### Installation via pip

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
df = pd.read_csv(path_of_folder + tabular_csv_data_name + '.csv', index_col=0, low_memory=False)
####################################
#### Data Preprocessing ####
####################################
kg_path = path_of_folder + tabular_csv_data_name
pipe = Pipeline([('createkg', KGCreator(path=kg_path)),
                 ('embeddings', ApplyKGE(params={'kge': 'Conve',  # Distmult,Complex,Tucker,Hyper, Conve
                                                 'embedding_dim': 10,
                                                 'batch_size': 256,
                                                 'num_epochs': 10}))])

model = pipe.fit_transform(X=df.select_dtypes(include='category'))
print(model)

```
