[![girth-tests Actions Status](https://github.com/eribean/girth/workflows/girth-tests/badge.svg)](https://github.com/eribean/girth/actions)
[![codecov](https://codecov.io/gh/eribean/girth/branch/master/graph/badge.svg?token=M7QW1P6V6X)](https://codecov.io/gh/eribean/girth)
[![CodeFactor](https://www.codefactor.io/repository/github/eribean/girth/badge)](https://www.codefactor.io/repository/github/eribean/girth)
[![PyPI version](https://badge.fury.io/py/girth.svg)](https://badge.fury.io/py/girth)
![PyPI - Downloads](https://img.shields.io/pypi/dm/girth)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# <ins>**G**</ins>eorgia Tech <ins>**I**</ins>tem <ins>**R**</ins>esponse <ins>**Th**</ins>eory

[![GIRTH](https://eribean.github.io/girth/featured-background_hubf3811d606e709c4b8d3b39f7338865e_285315_960x540_fill_q75_catmullrom_top.jpg)](https://eribean.github.io/girth/)

Girth is a python package for estimating item response theory (IRT) parameters.  In addition, synthetic IRT data generation is supported. Below is a list of available functions, for more information visit the GIRTH [homepage](https://eribean.github.io/girth/).

Interested in Bayesian Models? Check out [girth_mcmc](https://github.com/eribean/girth_mcmc). It provides markov chain and variational inference estimation methods.

**Dichotomous Models**
1. Rasch Model
   * Joint Maximum Likelihood
   * Conditional Likelihood
   * Marginal Maximum Likelihood
2. One Parameter Logistic Models
   * Joint Maximum Likelihood
   * Marginal Maximum Likelihood
3. Two Parameter Logistic Models
   * Joint Maximum Likelihood
   * Marginal Maximum Likelihood
   * Mixed Expected A Prior / Marginal Maximum Likelihood
3. Three Parameter Logistic Models
   * Marginal Maximum Likelihood (No Optimization and Minimal Support)

**Polytomous Models**
1. Graded Response Model
   * Joint Maximum Likelihood
   * Marginal Maximum Likelihood
   * Mixed Expected A Prior / Marginal Maximum Likelihood
2. Partial Credit Model
   * Joint Maximum Likelihood
   * Marginal Maximum Likelihood
3. Graded Unfolding Model
   * Marginal Maximum Likelihood

**Ablity Estimation**
1. Dichotomous
   * Marginal Likelihood Estimation
   * Maximum a Posteriori Estimation
   * Expected a Posteriori Estimation
2. Polytomous
   * Expected a Posteriori Estimation

**Supported Synthetic Data Generation**
1. Rasch / 1PL Models Dichotomous Models
2. 2 PL Dichotomous Models
3. 3 PL Dichotomous Models
4. Graded Response Model Polytomous
5. Partial Credit Model Polytomous
6. Graded Unfolding Model Polytomous
7. Multidimensional Dichotomous Models


## Installation
Via pip
```
pip install girth --upgrade
```

From Source
```
pip install . -t $PYTHONPATH --upgrade
```

## Dependencies
We recommend the anaconda environment which can be installed
[here](https://www.anaconda.com/distribution/)

* Python 3.7  
* Numpy  
* Scipy
* Numba

## Usage
```python
import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import twopl_mml

# Create Synthetic Data
difficulty = np.linspace(-2.5, 2.5, 10)
discrimination = np.random.rand(10) + 0.5
theta = np.random.randn(500)

syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, theta)

# Solve for parameters
estimates = twopl_mml(syn_data)

# Unpack estimates
discrimination_estimates = estimates['Discrimination']
difficulty_estimates = estimates['Difficulty']
```

### Missing Data
Missing data is supported with the `tag_missing_data` function.
```python
from girth import tag_missing_data
from girth import twopl_mml

# import data (you supply this function)
my_data = import_data(filename)

# Assume its dichotomous data with True -> 1 and False -> 0
tagged_data = tag_missing_data(my_data, [0, 1])

# Run Estimation
results = twopl_mml(tagged_data)
```

### Standard Errors
GIRTH does not use typical hessian based optimization routines and, therefore, *currently* 
has limited support for standard errors. Confidence Intervals based on bootstrapping are
supported but take longer to run. Missing Data is supported in the bootstrap function as well.

The bootstrap does not support the 3PL IRT Model or the GGUM.

```python
from girth import twopl_mml, standard_errors_bootstrap

# import data (you supply this function)
my_data = import_data(filename)

results = standard_errors_bootstrap(my_data, twopl_mml, n_processors=4,
                                    bootstrap_iterations=1000)

print(results['95th CI']['Discrimination'])                                    
```

## Unittests

**pytest** with coverage.py module
```
pytest --cov=girth --cov-report term
```

**nose** with coverage.py module
```
nosetests --with-coverage --cover-package=girth testing/
```

## Contact

Ryan Sanchez  
ryan.sanchez@gofactr.com

## License

MIT License

Copyright (c) 2021 Ryan C. Sanchez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
