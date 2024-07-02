# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from os import path
# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# packages required (replaces requirements.txt)
required = [
    'pandas>2.0',
    'matplotlib==3.6.1',
    'scikit-learn==1.5.0',
    'scipy==1.12.0',
    'jax==0.4.19',
    'jaxlib==0.4.19',
    'lightweight_mmm==0.1.9',
    'plotly==5.22.0',
    'optuna==3.6.1',
    "jinja2==3.0.1",
    # if running in a Jupyter Notebook env.
    "nbformat",
    "ipywidgets"
]

setup(
    name="mmx",
    version="0.1",
    package_dir={'': 'src'},
    packages=find_packages(),
    py_modules=['mmx'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
)
