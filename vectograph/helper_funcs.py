import os
def apply_PYKE(t):
    g, path ,params= t
    os.system("python PYKE/execute.py --kg_path {0} --embedding_dim {1} --eval True".format(path,params['embedding_dim']))
