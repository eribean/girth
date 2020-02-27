import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import rauch_approx, onepl_approx, twopl_approx
from girth import rauch_separate, onepl_separate, twopl_separate
from girth import rauch_full, onepl_full, twopl_full
from girth import rauch_jml, onepl_jml, twopl_jml
from girth import rasch_conditional













## TO ADD UNIT TESTS, CURRENTLY MAKINNG SURE
## THINGS IMPORT AND RUN




if __name__ == '__main__':
    difficuly = np.linspace(-2.5, 2.5, 10)
    discrimination = np.random.rand(10) + 0.5
    theta = np.random.randn(500)

    syn_data = create_synthetic_irt_dichotomous(difficuly, discrimination, theta)

    print('Testing Approximate Methods')
    rauch_approx(syn_data)
    onepl_approx(syn_data)
    twopl_approx(syn_data)

    print('Testing Separate Methods')
    rauch_separate(syn_data)
    onepl_separate(syn_data)
    twopl_separate(syn_data)

    print('Testing Full Methods')
    rauch_full(syn_data)
    onepl_full(syn_data)
    twopl_full(syn_data)

    print('Testing Conditional Methods')
    rasch_conditional(syn_data)

    print('Testing Joint Methods')
    rauch_jml(syn_data)
    onepl_jml(syn_data)
    twopl_jml(syn_data)
