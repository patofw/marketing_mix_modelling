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

**NOTE**: Scispacy can give errors when installing in MacOS M1, M2 or M3 machines. In that case, first execute:
`conda install nmslib`. In other operating systems this might not be required.

### Build the module

First, install the build package. This will allow you to build the Python Module and install its dependencies. 

`python -m pip install build`

or

`pip install --upgrade build`

then, Build the `medical-nlp` module using: 

`python -m build`

Finally, install all dependencies that are in the `setup.py` file with 

`pip install .`

------------------------
**NOTE**

For MACOS Silicon (M1, M2 or M3), before running `pip install .` you need to install `matplotlib` separately with:

`conda install -c conda-forge matplotlib`


-------------------------
After this, you can run 

`pip install .`

