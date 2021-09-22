import unittest  # pylint: disable=cyclic-import

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import rasch_mml, onepl_mml, twopl_mml

from girth import create_synthetic_irt_polytomous
from girth import grm_mml, pcm_mml, gum_mml

from girth import create_synthetic_mirt_dichotomous
from girth import multidimensional_grm_mml, multidimensional_twopl_mml
from girth.multidimensional_mml_methods import _build_einsum_string


class TestMMLRaschMethods(unittest.TestCase):

    # REGRESSION TESTS

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        rng = np.random.default_rng(5578134322131629)

        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.12
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)
        self.data = syn_data
        self.discrimination = discrimination
    
    def test_rasch_regression_mml(self):
        """Testing rasch separate methods."""
        syn_data = self.data.copy()
        output = rasch_mml(syn_data, self.discrimination)['Difficulty']
        expected_output = np.array([-1.438482, -0.723603,  0.029875,  0.666719,  1.406826])

        np.testing.assert_allclose(expected_output, output, atol=1e-3, rtol=1e-3)

    
    def test_rasch_close(self):
        """Testing rasch converging methods."""
        rng = np.random.default_rng(49843215977321216489)
        
        difficulty = np.linspace(-1.25, 1.25, 5)
        discrimination = 0.87
        thetas = rng.standard_normal(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)

        output = rasch_mml(syn_data, discrimination)['Difficulty']
        np.testing.assert_allclose(difficulty, output, atol=.1)


class TestMMLOnePLMethods(unittest.TestCase):

    # REGRESSION TESTS

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        rng = np.random.default_rng(8851224983218942)

        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.843
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)
        self.data = syn_data
        self.discrimination = discrimination
    
    def test_onepl_regression_mml(self):
        """Testing onepl separate methods."""
        syn_data = self.data.copy()
        output = onepl_mml(syn_data)
        
        expected_output = np.array([-1.336955, -0.785015,  0.028052,  
                                    0.699433,  1.539387])

        self.assertAlmostEqual(output['Discrimination'], 1.90677, places=4)
        np.testing.assert_allclose(expected_output, output['Difficulty'], 
                                   atol= 1e-3, rtol=1e-3)
    
    def test_onepl_close(self):
        """Testing onepl converging methods."""
        rng = np.random.default_rng(3215964845415)
        
        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.87
        thetas = rng.standard_normal(1000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)

        output = onepl_mml(syn_data)
        self.assertLess(np.abs(output['Discrimination'] - discrimination).max(), 0.1)
        self.assertLess(np.abs(output['Difficulty'] - difficulty).max(), 0.2)


class TestMMLTwoPLMethods(unittest.TestCase):

    # REGRESSION TESTS

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        rng = np.random.default_rng(3215964845415)

        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = rng.uniform(0, 1, 5) + 0.5
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)
        self.data = syn_data
        self.discrimination = discrimination
    
    def test_twopl_regression_mml(self):
        """Testing twopl separate methods."""
        syn_data = self.data.copy()
        output = twopl_mml(syn_data)

        expected_discrimination = np.array([0.990038, 0.528739, 0.599066, 
                                            0.827415, 0.625773])
        expected_output = np.array([-1.509518, -0.985841,  0.108493,  
                                    1.011885,  1.456912])

        np.testing.assert_allclose(
            expected_discrimination, output['Discrimination'], atol = 1e-4, rtol=1e-5)
        np.testing.assert_allclose(expected_output, output['Difficulty'], 
                                   atol = 1e-3, rtol=1e-3)
    
    def test_twopl_close(self):
        """Testing twopl converging methods."""
        rng = np.random.default_rng(54564132132146548)

        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.5 + rng.uniform(0, 1, 10)
        thetas = rng.standard_normal(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas, seed=rng)

        output = twopl_mml(syn_data)
        self.assertLess(np.abs(output['Discrimination'] - discrimination).mean(), 0.1)
        self.assertLess(np.abs(output['Difficulty'] - difficulty).mean(), 0.1)


class TestMMLGradedResponseModel(unittest.TestCase):
    """Tests the marginal maximum likelihood for GRM."""
    
    def test_graded_large_participant(self):
        """Regression Testing graded response model with large N."""
        rng = np.random.default_rng(37591254953128)
        
        difficulty = np.sort(rng.standard_normal((5, 4)), axis=1)
        discrimination = 0.5 + rng.uniform(0, 1, 5)
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, seed=rng)

        estimated_parameters = grm_mml(syn_data, {"use_LUT": False})

        # Regression test
        expected_discrimination = np.array([0.618059, 0.85724 , 0.933731, 0.783617, 0.768444])

        expectected_difficulty = np.array([[-0.79897775, -0.59202128,  0.49595011,  1.59808004],
                                           [-1.03595851, -0.19882228, -0.00903988,  0.33531401],
                                           [-0.8862972,  -0.45312044,  0.46180153,  2.47818242],
                                           [-1.12156704, -0.27140501,  0.21297667,  0.84264451],
                                           [-1.84418695, -1.66922269, -0.34500204,  0.29536871]])

        np.testing.assert_allclose(
            estimated_parameters['Discrimination'], expected_discrimination, 
            atol = 1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            estimated_parameters['Difficulty'], expectected_difficulty,
            atol = 1e-3, rtol=1e-3)
    
    def test_graded_small_participant(self):
        """Regression Testing graded response model with small N."""
        rng = np.random.default_rng(5378912)
        
        difficulty = np.sort(rng.standard_normal((5, 4)), axis=1)
        discrimination = 0.5 + rng.uniform(0, 1, 5)
        thetas = rng.standard_normal(101)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, seed=rng)

        estimated_parameters = grm_mml(syn_data)

        # Regression test
        expected_discrimination = np.array([1.395028, 1.362249, 1.245826, 1.062985, 0.353289])

        expectected_difficulty = np.array([[-2.87402313, -0.74836181,  1.40010846,  2.06246307],
                                           [-0.17760531,  0.94242998,  1.68052614,  np.nan],
                                           [-1.56364887, -0.66169962, -0.484627,   -0.06228502],
                                           [-0.39390175,  0.16104483,  1.15843314,  np.nan],
                                           [-2.50886987, -2.37244682, -1.59729314,  0.05772045]])

        np.testing.assert_allclose(
            estimated_parameters['Discrimination'], expected_discrimination, 
            atol = 1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            estimated_parameters['Difficulty'], expectected_difficulty, 
            atol = 1e-3, rtol=1e-3)
    
    def test_graded_response_model_close(self):
        """Regression Testing graded response model with large N."""
        rng = np.random.default_rng(32425613463421)
        difficulty = np.sort(rng.standard_normal((10, 4)), axis=1)
        discrimination = 0.5 + rng.uniform(0, 1, 10) + 0.5
        thetas = rng.standard_normal(1000)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, seed=rng)

        estimated_parameters = grm_mml(syn_data, {"use_LUT": False})

        rmse = np.sqrt(
            np.square(estimated_parameters['Discrimination'] - discrimination).mean())
        self.assertLess(rmse, .1)

        rmse = np.sqrt(np.square(estimated_parameters['Difficulty'] - difficulty).mean())
        self.assertLess(rmse, .1)
    
    def test_graded_response_LUT_vs_NOLUT(self):
        """Testing LUT give answer close to NO_LUT."""
        rng = np.random.default_rng(23015798234751908)
        difficulty = np.sort(rng.standard_normal((5, 4)), axis=1)
        discrimination = 0.5 + rng.uniform(0, 1, 5)
        thetas = rng.standard_normal(600)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, seed=rng)

        estimated_parameters_NOLUT = grm_mml(syn_data, {"use_LUT": False})
        estimated_parameters_LUT = grm_mml(syn_data, {"use_LUT": True})

        np.testing.assert_allclose(estimated_parameters_LUT['Difficulty'],
                                   estimated_parameters_NOLUT['Difficulty'], 
                                   atol=1e-3, rtol=1e-3)

        np.testing.assert_allclose(estimated_parameters_LUT['Discrimination'],
                                   estimated_parameters_NOLUT['Discrimination'], 
                                   atol=1e-3, rtol=1e-3)
    
    
    def test_graded_response_latent_distribution_estimate(self):
        """Testing LUT give answer close to NO_LUT."""
        rng = np.random.default_rng(984215958742346782)
        difficulty = np.sort(rng.standard_normal((20, 4)), axis=1)
        discrimination = 0.75 + rng.uniform(0, 1, 20)
        thetas = rng.standard_normal(1000)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas, seed=rng)

        estimated_parameters_ED = grm_mml(syn_data, {"use_LUT": True, 
                                                     "estimate_distribution": True,
                                                     "number_of_samples": 5})
        estimated_parameters_NED = grm_mml(syn_data, {"use_LUT": True, 
                                                      "estimate_distribution": False})

        # Just make sure it runs!
        np.testing.assert_allclose(estimated_parameters_ED['Difficulty'],
                                   estimated_parameters_NED['Difficulty'], 
                                   atol=.05, rtol=1e-3)

        np.testing.assert_allclose(estimated_parameters_ED['Discrimination'],
                                   estimated_parameters_NED['Discrimination'], 
                                   atol=.05, rtol=1e-3)                                   

class TestMMLPartialCreditModel(unittest.TestCase):
    """Tests the marginal maximum likelihood for GRM."""

    @classmethod
    def setUp(self):
        rng = np.random.default_rng(893476329804797)
        self.difficulty = rng.standard_normal((10, 4))
        self.discrimination = 0.5 + rng.uniform(0, 1, 10)
        thetas = rng.standard_normal(1000)
        thetas_smol = thetas[:500].copy()

        self.discrimination_smol = self.discrimination[:5].copy()
        self.difficulty_smol = self.difficulty[:5, :].copy()

        self.syn_data_smol = create_synthetic_irt_polytomous(self.difficulty_smol,
                                                             self.discrimination_smol,
                                                             thetas_smol,
                                                             model='PCM',
                                                             seed=45270983475)

        self.syn_data_larg = create_synthetic_irt_polytomous(self.difficulty,
                                                             self.discrimination,
                                                             thetas,
                                                             model='PCM',
                                                             seed=45270983475)

        self.syn_data_mixed = create_synthetic_irt_polytomous(self.difficulty_smol[:, 1:],
                                                              self.discrimination_smol,
                                                              thetas,
                                                              model='PCm',
                                                              seed=45270983475)
    
    def test_pcm_gets_better(self):
        """Testing mml partial credit model improves with parameters."""
        output_smol = pcm_mml(self.syn_data_smol)
        output_large = pcm_mml(self.syn_data_larg)

        def rmse(expected, result):
            return np.sqrt(np.nanmean(np.square(expected - result)))

        rmse_smol = rmse(output_smol['Discrimination'], self.discrimination_smol)
        rmse_large = rmse(output_large['Discrimination'][:5], self.discrimination_smol)

        self.assertLess(rmse_large, rmse_smol)
        self.assertAlmostEqual(rmse_large, 0.047146750, places=4)

        # Regression Tests
        expected_discr = np.array([0.580641, 0.933542, 0.545533, 
                                   1.384329, 1.002218, 1.289025,
                                   0.666627, 0.935574, 1.075436, 0.805632])

        expected_diff = np.array([
            [-0.73888022,  0.48658471,  0.54432479, -0.20845413],
            [-1.60420276,  1.51068354,  0.40853116,  0.96107097],
            [ 0.74859512, -0.8266682,  -0.3281444,  -0.59730418],
            [-1.26196949,  1.21638843,  0.81767559,  0.14749641],
            [-1.25641728, -1.91429535,  0.29183112,  1.42147402],
            [ 0.10975361,  0.37555775, -0.61898549, -0.64545493],
            [-0.09936281, -0.81053536,  2.57210188, -0.15291919],
            [ 1.82337828, -3.48583283, -0.46336348, -0.61231518],
            [-0.52381878, -0.04939606,  1.46079265, -0.97823618],
            [ 0.14933984,  1.50334098,  0.07215466,  0.34340747]            
        ])

        np.testing.assert_allclose(
            expected_discr, output_large['Discrimination'], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            expected_diff, output_large['Difficulty'], atol=1e-3, rtol=1e-3)
    
    def test_pcm_mixed_difficulty_length(self):
        """Testing response set with different difficulty lengths."""
        syn_data = self.syn_data_larg.copy()
        syn_data[:5, :] = self.syn_data_mixed

        expected_diff = np.array([
            [ 0.21026932,  0.52425421, -0.18010127, np.nan],
            [ 1.41313701,  0.4436859,   0.93681588, np.nan],
            [-0.42425069, -0.59080182, -0.53292842, np.nan],
            [ 1.27658341,  0.89566005,  0.01640591, np.nan],
            [-2.06755279,  0.30266522,  1.411641,   np.nan],
            [ 0.12881731,  0.38916135, -0.63986058, -0.67991199],
            [-0.10296058, -0.81505543,  2.57057425, -0.15957945],
            [ 1.79151261, -3.46598156, -0.46742849, -0.60743497],
            [-0.533436,  -0.05599013,  1.43555693, -0.94565827],
            [ 0.13935706,  1.48691241,  0.07553486,  0.34733299]
        ])

        expected_discr = np.array([0.58303313, 0.98298398, 0.5275932, 1.33180649, 
                                   0.99874869, 1.25343601, 0.66599838, 0.94446462, 
                                   1.09705009, 0.81375833])
        output = pcm_mml(syn_data)

        np.testing.assert_allclose(
            expected_diff, output['Difficulty'], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            expected_discr, output['Discrimination'], atol=1e-3, rtol=1e-3)


class TestMMLGradedUnfoldingModel(unittest.TestCase):
    """Tests the marginal maximum likelihood for GUM."""

    # Smoke / Regression Tests
    
    def test_unfolding_run(self):
        """Testing the unfolding model runs."""
        rng = np.random.default_rng(73424353456772)
        difficulty = -rng.uniform(0, 1.5, (10, 2))
        delta = rng.uniform(0, 0.5, (10, 1))
        discrimination = rng.uniform(0, 1, 10) + 0.5
        thetas = rng.standard_normal(300)

        betas = np.c_[difficulty, np.zeros((10, 1)), -difficulty[:, ::-1]]
        betas += delta
        syn_data = create_synthetic_irt_polytomous(betas, discrimination,
                                                   thetas, model='GUM',
                                                   seed=rng)
        delta_sign = (0, 1)
        result = gum_mml(syn_data, delta_sign=delta_sign, options={'max_iteration': 100})

        rmse_discrimination = np.sqrt(np.square(discrimination.squeeze() - 
                                                result['Discrimination']).mean())
        rmse_delta = np.sqrt(np.square(delta.squeeze() - 
                                       result['Delta']).mean())
        rmse_tau = np.sqrt(np.square(difficulty - 
                                     result['Tau']).mean())

        self.assertAlmostEqual(rmse_discrimination, 0.329984238, places=4)
        self.assertAlmostEqual(rmse_delta, 0.54465562, places=3)
        self.assertAlmostEqual(rmse_tau, 0.1778829854, places=3)

    def test_unfolding_specify_negative_ndx(self):
        """Testing specifying a negative index."""
        rng = np.random.default_rng(34572348572012334)
        difficulty = -rng.uniform(0, 1.5, (10, 2))
        delta = rng.uniform(0, 0.5, (10, 1))
        discrimination = rng.uniform(0, 1, 10) + 0.5
        thetas = rng.standard_normal(300)

        betas = np.c_[difficulty, np.zeros((10, 1)), -difficulty[:, ::-1]]
        betas += delta
        syn_data = create_synthetic_irt_polytomous(betas, discrimination,
                                                   thetas, model='GUM',
                                                   seed=rng)

        delta_sign = (2, np.sign(delta[2]))
        result = gum_mml(syn_data, delta_sign=delta_sign, 
                         options={'max_iteration': 100})

        self.assertEqual(np.sign(result['Delta'][2]), delta_sign[1])
        self.assertNotEqual(np.sign(result['Delta'][2]), -delta_sign[1])


class TestMultiDimensionalIRT(unittest.TestCase):
    """Test fixture for multidimensional IRT."""
    
    def test_einsum_string(self):
        """Test building the einsum string."""
        einString = _build_einsum_string(2)
        self.assertEqual(einString, "a, b -> ab")

        einString = _build_einsum_string(5)
        self.assertEqual(einString, "a, b, c, d, e -> abcde")
        
        with self.assertRaises(ValueError):
            _build_einsum_string(12)

    def test_multidimensional_2pl(self):
        """Testing Multidimensional 2PL Model."""
        rng = np.random.default_rng(93209819094946739803948765)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = rng.uniform(-2, 2, (5, 2))
        thetas = rng.standard_normal((2, 500))
        syn_data = create_synthetic_mirt_dichotomous(difficulty, 
                                                     discrimination, 
                                                     thetas, seed=rng)
        
        results = multidimensional_twopl_mml(syn_data, 2, 
                                            {'quadrature_n': 15,
                                             'max_iteration': 750})

        # Regression Tests
        expected_discrimination = np.array([
            [-1.85670663,  2.72354644],
            [-1.33970775, -0.14605692],
            [ 0.78865943,  1.48472356],
            [ 1.29734554,  1.26802896],
            [ 0.56405506,  0.        ]])
        
        expected_LL = -1448.7247702457792

        expected_difficulty = np.array([-2.74171551, -0.92506653, 0.03592453,  
                                        0.96030869, 1.49056813])

        np.testing.assert_allclose(expected_LL, results['LL'], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(np.abs(expected_discrimination), 
                                   np.abs(results['Discrimination']), atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(expected_difficulty, results['Difficulty'], atol=1e-3, rtol=1e-3)

    def test_multidimensional_grm(self):
        """Testing Multidimensional GRM Model."""
        rng = np.random.default_rng(93209819094946739803948765)
        difficulty = -1*np.sort(rng.uniform(-1.5, 1.5, (5, 3)), axis=1)
        discrimination = rng.uniform(-2, 2, (5, 3))
        thetas = rng.standard_normal((3, 500))

        syn_data = create_synthetic_irt_polytomous(difficulty, 
                                                   discrimination, 
                                                   thetas, model='grm_md',
                                                   seed=rng)
        
        results = multidimensional_grm_mml(syn_data, 3, 
                                            {'quadrature_n': 15,
                                             'max_iteration': 10})

        # Regression Tests
        expected_discrimination = np.array([
            [-1.09835568, -1.28947079, -2.34486106],
            [-1.03379657, -2.06350004, -0.38485667],
            [-2.41051063,  2.09253819,  0.18879694],
            [ 0.53951431,  0.74297555,  0.        ],
            [ 2.51349768,  0.        ,  0.        ]])
        
        expected_LL = -2696.852894303332

        expected_difficulty = np.array([
            [ 2.42581536,  1.18280897, -0.14802899],
            [ 0.59854883,  0.48006261, -0.68823203],
            [ 0.56426239, -0.45650437, -0.89701715],
            [ 0.12291629, -0.22732149, -0.63694442],
            [ 1.00906448, -1.30930484, -1.34350456]])

        np.testing.assert_allclose(expected_LL, results['LL'], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(np.abs(expected_discrimination), 
                                   np.abs(results['Discrimination']), atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(expected_difficulty, results['Difficulty'], atol=1e-3, rtol=1e-3)

        with self.assertRaises(AssertionError):
            multidimensional_grm_mml(syn_data, 1)


if __name__ == '__main__':
    unittest.main()
