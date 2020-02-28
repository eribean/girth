# <ins>**G**</ins>eorgia Tech <ins>**I**</ins>tem <ins>**R**</ins>esponse <ins>**Th**</ins>eory Package
The GIRTh package is intended to be a python module implementing a broad swath of item response theory parameter estimation packages.

## Dependencies

Python 3.7

Numpy

Scipy

We use the anaconda environment which can be installed
Download [here](https://www.anaconda.com/distribution/)

## Installation
```
python setup.py install --prefix=path/to/your/installation
```

## Usage
```python
import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import twopl_separate

# Create Synthetic Data
difficuly = np.linspace(-2.5, 2.5, 10)
discrimination = np.random.rand(10) + 0.5
theta = np.random.randn(500)

syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination, theta)

# Solve for parameters
estimates = twopl_separate(syn_data)

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

## Contact

Ryan Sanchez
rsanchez44@gatech.edu
