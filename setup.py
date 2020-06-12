import os

from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='vectograph',
    description='A set of python modules for applying knowledge graph embedding on tabular data',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'test.*','examples.*')),
    install_requires=['scikit-learn==0.22.1',
                      'pandas>=1.0.3',
                      'rdflib>=5.0.0',
                      'matplotlib>=3.2.1'],
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", ],
    python_requires='>=3.6',
    long_description=read('README.md'),
    long_description_content_type="test/markdown",
)