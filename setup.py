from setuptools import setup, find_packages
with open('README.md', 'r') as fh:
    long_description = fh.read()
setup(
    name='vectograph',
    description='A set of python modules for applying knowledge graph embedding on tabular data',
    version='0.0.2',
    packages=find_packages(exclude=('tests', 'gi.*','examples.*')),
    install_requires=['scikit-learn==0.22.1',
                      'pandas>=1.0.3',
                      'torch'],
    author='Caglar Demir',
    author_email='caglardemir8@gmail.com',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", ],
    python_requires='>=3.6',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
