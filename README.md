# TSP-IT
An Independence Test based on Data-Driven Tree-Structured Representations.

## Contents
This repository includes:
- A Python wrapper to C++ implementation of the histogram-based mutual information estimator based on data-driven tree-structured partitions (TSP) presented by [[Silva et al. (2012)](https://arxiv.org/pdf/2110.14122.pdf)].
- A python implementation of the independence test based on the data-driven tree-structured partitions (TSP) presented by [[Gonzalez et al. (2021)](https://arxiv.org/pdf/2110.14122.pdf)] (including all the experiments developed).

## Requirements
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page).
* [CMake](https://cmake.org/).
* [Boost](https://www.boost.org/).
* [Python 3](https://www.python.org/).
* [NumPy](https://numpy.org/).

## Installation
### Clone
- Clone this repo to your local machine using `https://github.com/mvidela31/TSP-MI`
### Setup
- Set the NumPy library path on line 9 of the file `./src/CMakeLists.txt`. For example:
```CMakeLists
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} /home/usr/anaconda3/lib/python3.7/site-packages/numpy/core/include)
```
### Compilation
```Shell
$ cd ./src/build
$ cmake ..
$ make
```
- The python package `TSP.py` it's now builded! Now you can run the Jupyter notebooks from the folder `experiments`.

## Usage
- Python package `TSP.py` example of use.
```Python
import sys
import numpy as np
# Set the TSP.py package path
sys.path.insert(1, '../src/build')
# Then import the package
from TSP import TSP

# Initialize NumPy arrays of shape (n_samples, dim) with random variables samples in Fortran-contiguous memory order.
X = np.array([...], order='F')
Y = np.array([...], order='F')

# Initialize the TSP with its parameters
l_bn = 0.167
w_bn = 5e-2
reg_lambda = 2.5e-5
tsp = TSP(l_bn, w_bn, reg_lambda)

# Set observations to the estimator
tsp.grow(X, Y)

# Get TSP size and estimated mutual information
size = tsp.size()
emi = tsp.emi()

# Regularize the TSP mutual information estimator
tsp.regularize()

# Get the new values of size and estimated mutual information
new_size = tsp.size()
new_emi = tsp.emi()
```

## References
[[1](https://arxiv.org/pdf/2110.14122.pdf)]  Gonzalez, M. E., Silva, J. F., Videla, M., & Orchard, M. E. (in press). **Data-Driven Representations for Testing Independence: Modeling, Analysis and Connection with Mutual Information Estimation**. *IEEE Transactions on Signal Processing*.

[[2](https://sail.usc.edu/publications/files/silva_tit_2012.pdf)] Silva, J. F., & Narayanan, S. (2012). **Complexity-regularized tree-structured partition for mutual information estimation**. *IEEE transactions on information theory, 58*(3), 1940-1952.
