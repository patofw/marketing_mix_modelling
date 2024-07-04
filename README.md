# marketing_mix_modelling
Pharma based Marketing Mix Modelling tutorial.


## Installation 

**NOTE**: Using a Bash terminal is recommended for the set up. 

It is assumed Python and Conda have been installed. Also worth noting that SciSpacy requires python==3.11 to work in MacOS M1, M2 and M3 machines. 

Make sure you have the latest version of pip install: `python -m pip install --upgrade pip`

I highly recommend building a new virtual environment. You can use conda for example (replace <ENVNAME> by your desired v-env name): 

`conda create -n <ENVNAME> python==3.12`

activate your virtual environment after you created it.

`conda activate <ENVNAME>`

Build the module. This will allow to import methods and classes from the package seamlessly.


### Build the module

First, install the build package. This will allow you to build the Python Module and install its dependencies. 

`python -m pip install build`

or

`pip install --upgrade build`

then, Build the `mmx` module using: 

`python -m build`

---
> **NOTE**:

>For MACOS Silicon (M1, M2 or M3), before running `pip install .` you need to install & `jinja2` separately with:

> `conda install -c conda-forge jinja2`
---

After this, you can continue with the normal installation process.



Finally, install all dependencies that are in the `setup.py` file with 

`pip install -e .`




## Exercises.

This tutorial has 2 main notebooks. The [mmx_linear_model_example](./analysis/mmx_linear_model_example.ipynb) notebook is recommended to be studied first, followed by the [mmx_bayesian_model_example](./analysis/mmx_bayesian_model_example.ipynb) one. 

Both resources give you the fundamental understanding of marketing mix modelling and how it can be applied in the pharmaceutical industry (or any other industry).

---

## About the data sets.

This repo has 3 datasets that can be used in different marketing mix examples. Data is synthetic, but it realistically follows the sales patterns for __off-the-counter (OTC)__ products.

1. [cough_and_cold_sales:](./data/cough_and_cold_sales.csv) This data set resembles the sales for a traditional cough and cold product that is sold OTC in pharmacies in a particular European market. As it is expected, cough and cold sales grow during winter and decrease in summer. In addition to the sales column, normalized marketing spends and relevant external factors are included. Likewise, the sales are realistically impacted by the COVID Pandemic. 
2. [pain_killer_sales](./data/pain_killer_sales.csv): This data set is similar to the one mentioned above, however it resembles the sales of pain killers, which are much less sensitive to seasonality. This dataset should be loaded and analysed together with its [cost data](./data/pain_killer_cost.csv). The latter gives an aggregated cost of the marketing spends. 
3. [simple_data_sample](./data/simple_data_sample.csv): This is a simple dataset that can be used in any marketing mix tutorial.


## Running in Colab and "quick install"

To install the `mmx` package in colab, simply open a new [Colab Notebook](https://colab.research.google.com/) and in the first cell run

```bash

!pip install git+https://github.com/patofw/marketing_mix_modelling.git
```

You can verify the installation with 

```python

import mmx

```

**WARNING** -> MACOS Silicon systems may encounter dependency issues. 

