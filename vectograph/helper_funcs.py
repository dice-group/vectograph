import os
def apply_PYKE(t):
    g, path = t
    os.system("python PYKE/execute.py --K 20 --kg_path {0}".format(path))
