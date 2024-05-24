# -*- coding: utf-8 -*-
from setuptools import setup
from os import path
# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# packages required (replaces requirements.txt)
required = [
    'pandas>2.0',
    'scikit-learn==1.5.0',
    'matplotlib',
    'lightweight_mmm==0.1.9'
]

setup(
    name="mmx",
    version="0.1",
    packages=["src", "src.mmx"],
    py_modules=['mmx'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    # dependency_links=[] add model links if desired here
)
