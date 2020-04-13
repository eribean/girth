![CircleCI](https://circleci.com/gh/eribean/girth.svg?style=shield)
![codecov.io](https://codecov.io/gh/eribean/girth/coverage.svg?branch=master)
[![CodeFactor](https://www.codefactor.io/repository/github/eribean/girth/badge)](https://www.codefactor.io/repository/github/eribean/girth)
[![PyPI version](https://badge.fury.io/py/girth.svg)](https://badge.fury.io/py/girth)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# <ins>**G**</ins>eorgia Tech <ins>**I**</ins>tem <ins>**R**</ins>esponse <ins>**Th**</ins>eory Package
Girth is a python package for estimating item response theory (IRT) parameters.  In addition, synthetic IRT data generation is supported. Below is a list of available functions, for more information visit the GIRTH [homepage](https://eribean.github.io/girth/).


**Dichotomous Models**
1. Rasch Model
   * Joint Maximum Likelihood
   * Conditional Likelihood
   * Marginal Maximum Likelihood
2. One / Two Parameter Logistic Models
   * Joint Maximum Likelihood
   * Marginal Maximum Likelihood

**Polytomous Models**
1. Graded Response Model
   * Joint Maximum Likelihood
   * Marginal Maximum Likelihood
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
python setup.py install --prefix=path/to/your/installation
```

## Usage
```python
import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import twopl_mml

# Create Synthetic Data
difficuly = np.linspace(-2.5, 2.5, 10)
discrimination = np.random.rand(10) + 0.5
theta = np.random.randn(500)

syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination, theta)

# Solve for parameters
estimates = twopl_mml(syn_data)

# Unpack estimates
discrimination_estimates = estimates[0]
difficulty_estimates = estimates[1]
```

## Unittests

**Without** coverage.py module
```
nosetests testing/
```

**With** coverage.py module
```
nosetests --with-coverage --cover-package=girth testing/
```

## Dependencies

* Python 3.7  
* Numpy  
* Scipy  

We use the anaconda environment which can be installed
Download [here](https://www.anaconda.com/distribution/)

## Contact

Ryan Sanchez  
rsanchez44@gatech.edu

## License

MIT License

Copyright (c) 2020 Ryan Sanchez

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
