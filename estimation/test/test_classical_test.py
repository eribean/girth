import unittest

import numpy as np

from girth.synthetic import create_synthetic_irt_polytomous
from girth import classical_test_statistics


class TestClassicalStatistics(unittest.TestCase):
    """Test Fixture for classical test statistics."""

    def setUp(self):
        """Testing classical test statistics."""
        rng = np.random.default_rng(465234543084621232141567865641323)

        discrimnation = rng.uniform(-2, 2, (20, 2))
        thetas = rng.standard_normal((2, 1000))
        difficulty = -1 * np.sort(rng.standard_normal((20, 3))*.5, axis=1)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimnation, 
                                                   thetas, model='grm_md', seed=rng)
        self.data = syn_data

    def test_classical_statistics_polytomous(self):
        """Testing CTT polytomous."""
        # Run Classical Test Statistics (Polytomous)
        results = classical_test_statistics(self.data, start_value=1, stop_value=4)

        # Regression Test Compared with R package CTT (itemAnalysis, rBisML=TRUE)
        item_mean = np.array([2.649, 2.793, 2.611, 2.581, 2.551, 2.404, 2.533, 2.613, 2.127,
                              2.536, 2.425, 2.401, 2.762, 2.248, 2.573, 2.458, 2.549,
                              2.711, 2.263, 2.715])
        
        ptPolySerial = np.array([0.18519409, -0.26677081,  0.22701664,  0.17493246,  0.01857994, 
                                -0.19123726, 0.20895748, -0.18350961,  0.24555211, -0.04361278,  
                                0.38517507,  0.08073748, 0.32387954,  0.20053442,  0.11787938, 
                                -0.15320180, -0.15141016,  0.05840297, 0.36121673,  0.41722428])

        polySerial = np.array([0.22329727, -0.32313421,  0.26475102,  0.20711435,  0.01978431,
                              -0.21853317,0.24929137, -0.22308782,  0.29620018, -0.04983768,  
                               0.44225203,  0.09177238, 0.38307279,  0.24054010,  0.14778010, 
                               -0.18521153, -0.17929838,  0.07107441, 0.42084277,  0.49616586])

        alpha_if_deleted = np.array([0.3071786, 0.4239081, 0.2945200, 0.3088433, 0.3512205, 
                                     0.4029160, 0.2986372, 0.4051149, 0.2890540, 0.3685364, 
                                     0.2506303, 0.3348770, 0.2668133, 0.3016237, 0.3246973, 
                                     0.3982845, 0.3950056, 0.3410210, 0.2577887, 0.2341913])

        np.testing.assert_allclose(results['Mean'], item_mean, atol=1e-4)
        np.testing.assert_allclose(results['Item-Score Correlation'], ptPolySerial, atol=5e-4)
        np.testing.assert_allclose(results['Polyserial Correlation'], polySerial, atol=5e-4)
        np.testing.assert_allclose(results['Cronbach Alpha'], alpha_if_deleted, atol=5e-4)



    def test_classical_statistics_dichotomous(self):
        """Testing CTT dichotomous."""        
        # Run Classical Test Statistics (Dichotomous)
        results = classical_test_statistics(self.data > 2, 
                                            start_value=0, stop_value=1)

        # Regression Test Compared with R package CTT (itemAnalysis, rBisML=TRUE)
        item_mean = np.array([0.507, 0.605, 0.532, 0.516, 0.478, 0.507, 0.513, 0.549, 0.367,
                              0.507, 0.522, 0.502, 0.572, 0.443, 0.509, 0.499, 0.499, 0.569, 
                              0.417, 0.579])
        
        ptPolySerial = np.array([0.13635945, -0.23690518,  0.19925177,  0.16125569,  
                                 0.01521503, -0.17498578, 0.20124621, -0.17413208,
                                 0.22529891, -0.04521061,  0.32988860,  0.06748577,
                                 0.29354804,  0.19119399,  0.09586074, -0.14640699, 
                                 -0.12523201,  0.05451945, 0.30543711,  0.38981897])

        polySerial = np.array([0.17025715, -0.29648870,  0.24760697, 0.20167912, 
                               0.01907977, -0.21813488, 0.24965703, -0.21828750,
                               0.28844678, -0.05666627,  0.40674749,  0.08458126,
                               0.36462512,  0.23865434, 0.11969521, -0.18380548,
                               -0.15640287,  0.06870646, 0.38111201, 0.47901615])

        alpha_if_deleted = np.array([0.2841804, 0.3854030, 0.2653823, 0.2767744, 0.3192645, 
                                     0.3711947, 0.2647135, 0.3706409, 0.2590128, 0.3361899, 
                                     0.2247302, 0.3043355, 0.2368803, 0.2679735, 0.2960959, 
                                     0.3636431, 0.3579882, 0.3079945, 0.2334289, 0.2066341])

        np.testing.assert_allclose(results['Mean'], item_mean, atol=1e-4)
        np.testing.assert_allclose(results['Item-Score Correlation'], ptPolySerial, atol=5e-4)
        np.testing.assert_allclose(results['Polyserial Correlation'], polySerial, atol=6e-4)
        np.testing.assert_allclose(results['Cronbach Alpha'], alpha_if_deleted, atol=6e-4)




if __name__ == "__main__":
    unittest.main()