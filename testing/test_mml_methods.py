import unittest  # pylint: disable=cyclic-import

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import rasch_mml, onepl_mml, twopl_mml
from girth import rasch_full, onepl_full, twopl_full

from girth import create_synthetic_irt_polytomous
from girth import grm_mml, pcm_mml, gum_mml


class TestMMLRaschMethods(unittest.TestCase):

    # REGRESSION TESTS

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(3)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.12
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)
        self.data = syn_data
        self.discrimination = discrimination

    def test_rasch_regression_mml(self):
        """Testing rasch separate methods."""
        syn_data = self.data.copy()
        output = rasch_mml(syn_data, self.discrimination)['Difficulty']
        expected_output = np.array([-1.324751, -0.814625, -0.082224,  0.658678,  1.470566])

        np.testing.assert_allclose(expected_output, output, atol=1e-3, rtol=1e-3)

    def test_rasch_regression_full(self):
        """Testing rasch full methods."""
        syn_data = self.data.copy()
        output = rasch_full(syn_data, self.discrimination)['Difficulty']
        expected_output = np.array([-1.32206195, -0.81438101, -0.0847999, 
                                     0.65460933,  1.4664586])

        np.testing.assert_allclose(expected_output, output, atol=1e-3, rtol=1e-3)

    def test_rasch_close(self):
        """Testing rasch converging methods."""
        np.random.seed(333)
        difficulty = np.linspace(-1.25, 1.25, 5)
        discrimination = 0.87
        thetas = np.random.randn(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = rasch_mml(syn_data, discrimination)['Difficulty']
        np.testing.assert_array_almost_equal(difficulty, output, decimal=1)


class TestMMLOnePLMethods(unittest.TestCase):

    # REGRESSION TESTS

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(873)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = 1.843
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)
        self.data = syn_data
        self.discrimination = discrimination

    def test_onepl_regression_mml(self):
        """Testing onepl separate methods."""
        syn_data = self.data.copy()
        output = onepl_mml(syn_data)
        
        expected_output = np.array([-1.376502, -0.648995, -0.039338,  0.77919 ,  1.386173])

        self.assertAlmostEqual(output['Discrimination'], 1.901730570, places=4)
        np.testing.assert_allclose(expected_output, output['Difficulty'], 
                                   atol= 1e-3, rtol=1e-3)

    def test_onepl_regression_full(self):
        """Testing onepl full methods."""
        syn_data = self.data.copy()
        output = onepl_full(syn_data)
        expected_output = np.array([-1.37825764, -0.64679736, -0.03537104, 
                                     0.78121678,  1.38471631])

        self.assertAlmostEqual(output['Discrimination'], 1.9019012, places=4)
        np.testing.assert_allclose(expected_output, output['Difficulty'], 
                                   atol= 1e-3, rtol=1e-3)

    def test_onepl_close(self):
        """Testing onepl converging methods."""
        np.random.seed(843)
        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.87
        thetas = np.random.randn(1000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = onepl_mml(syn_data)
        self.assertLess(np.abs(output['Discrimination'] - discrimination).max(), 0.1)
        self.assertLess(np.abs(output['Difficulty'] - difficulty).max(), 0.2)


class TestMMLTwoPLMethods(unittest.TestCase):

    # REGRESSION TESTS

    """Setup synthetic data."""

    def setUp(self):
        """Setup synthetic data for tests."""
        np.random.seed(247)
        difficulty = np.linspace(-1.5, 1.5, 5)
        discrimination = np.random.rand(5) + 0.5
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)
        self.data = syn_data
        self.discrimination = discrimination

    def test_twopl_regression_mml(self):
        """Testing twopl separate methods."""
        syn_data = self.data.copy()
        output = twopl_mml(syn_data)

        expected_discrimination = np.array([0.99981316, 1.86369226, 1.35526711, 
                                            0.52935723, 0.90899136])
        expected_output = np.array([-1.315087, -0.648258, -0.079815,  0.774052,  1.66791])

        np.testing.assert_allclose(
            expected_discrimination, output['Discrimination'], atol = 1e-4, rtol=1e-5)
        np.testing.assert_allclose(expected_output, output['Difficulty'], 
                                   atol = 1e-3, rtol=1e-3)

    def test_twopl_regression_full(self):
        """Testing twopl full methods."""
        syn_data = self.data.copy()
        output = twopl_full(syn_data)

        expected_discrimination = np.array([0.99979828, 1.86386639, 1.35529227, 
                                            0.5293589 , 0.90905802])
        expected_output = np.array([-1.31527794, -0.6482246 , -0.08031856, 
                                     0.77397527,  1.66750714])

        np.testing.assert_allclose(
            expected_discrimination, output['Discrimination'], atol = 1e-4, rtol=1e-5)
        np.testing.assert_allclose(expected_output, output['Difficulty'], 
                                   atol = 1e-3, rtol=1e-3)

    def test_twopl_close(self):
        """Testing twopl converging methods."""
        np.random.seed(43)
        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.5 + np.random.rand(10)
        thetas = np.random.randn(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = twopl_mml(syn_data)
        self.assertLess(np.abs(output['Discrimination'] - discrimination).mean(), 0.1)
        self.assertLess(np.abs(output['Difficulty'] - difficulty).mean(), 0.1)


class TestMMLGradedResponseModel(unittest.TestCase):
    """Tests the marginal maximum likelihood for GRM."""

    def test_graded_large_participant(self):
        """Regression Testing graded response model with large N."""
        np.random.seed(1944)
        difficulty = np.sort(np.random.randn(5, 4), axis=1)
        discrimination = np.random.rand(5) + 0.5
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        estimated_parameters = grm_mml(syn_data, {"use_LUT": False})

        # Regression test
        expected_discrimination = np.array([0.582914, 1.38349 , 0.879928, 1.173225, 1.471951])

        expectected_difficulty = np.array([[-1.23881672, -0.5340817,  0.06170343,  1.23881201],
                                           [-1.09279868, -0.76747967,
                                               0.44660955,  1.28909032],
                                           [-0.16828803,  0.23943693,
                                               0.76140209,  1.24435541],
                                           [-2.02935022, -0.70214267, -
                                               0.23281603,  1.27521818],
                                           [-1.47758497, -0.9050062,  0.0698804,  0.71286592]])

        np.testing.assert_allclose(
            estimated_parameters['Discrimination'], expected_discrimination, 
            atol = 1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            estimated_parameters['Difficulty'], expectected_difficulty,
            atol = 1e-3, rtol=1e-3)

    def test_graded_small_participant(self):
        """Regression Testing graded response model with small N."""
        np.random.seed(87)
        difficulty = np.sort(np.random.randn(5, 4), axis=1)
        discrimination = np.random.rand(5) + 0.5
        thetas = np.random.randn(101)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        estimated_parameters = grm_mml(syn_data)

        # Regression test
        expected_discrimination = np.array([1.724843, 0.393598, 0.828646, 0.937709, 1.567619])

        expectected_difficulty = np.array([
                                    [-0.1571773,   0.22757906,  0.33440129,  0.59312807],
                                    [ 1.22120566,  1.33164642,  2.26644668,  3.51066062],
                                    [-1.25491225, -1.06215245, -0.70060057,  0.87804437],
                                    [-1.26594612, -0.37869587,  2.2565641,       np.nan],
                                    [-1.36972366, -0.66190223, -0.50312446,      np.nan]])

        np.testing.assert_allclose(
            estimated_parameters['Discrimination'], expected_discrimination, 
            atol = 1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            estimated_parameters['Difficulty'], expectected_difficulty, 
            atol = 1e-3, rtol=1e-3)

    def test_graded_response_model_close(self):
        """Regression Testing graded response model with large N."""
        np.random.seed(6322)
        difficulty = np.sort(np.random.randn(10, 4), axis=1)
        discrimination = np.random.rand(10) + 0.5
        thetas = np.random.randn(1000)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        estimated_parameters = grm_mml(syn_data, {"use_LUT": False})

        rmse = np.sqrt(
            np.square(estimated_parameters['Discrimination'] - discrimination).mean())
        self.assertLess(rmse, .0966)

        rmse = np.sqrt(np.square(estimated_parameters['Difficulty'] - difficulty).mean())
        self.assertLess(rmse, .1591)

    def test_graded_response_LUT_vs_NOLUT(self):
        """Testing LUT give answer close to NO_LUT."""
        np.random.seed(489413)
        difficulty = np.sort(np.random.randn(5, 4), axis=1)
        discrimination = np.random.rand(5) + 0.5
        thetas = np.random.randn(600)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        estimated_parameters_NOLUT = grm_mml(syn_data, {"use_LUT": False})
        estimated_parameters_LUT = grm_mml(syn_data, {"use_LUT": True})

        np.testing.assert_allclose(estimated_parameters_LUT['Difficulty'],
                                   estimated_parameters_NOLUT['Difficulty'], 
                                   atol=1e-3, rtol=1e-3)

        np.testing.assert_allclose(estimated_parameters_LUT['Discrimination'],
                                   estimated_parameters_NOLUT['Discrimination'], 
                                   atol=1e-3, rtol=1e-3)

class TestMMLPartialCreditModel(unittest.TestCase):
    """Tests the marginal maximum likelihood for GRM."""

    @classmethod
    def setUp(self):
        np.random.seed(1944)
        self.difficulty = np.random.randn(10, 4)
        self.discrimination = np.random.rand(10) + 0.5
        thetas = np.random.randn(1000)
        thetas_smol = thetas[:500].copy()

        self.discrimination_smol = self.discrimination[:5].copy()
        self.difficulty_smol = self.difficulty[:5, :].copy()

        self.syn_data_smol = create_synthetic_irt_polytomous(self.difficulty_smol,
                                                             self.discrimination_smol,
                                                             thetas_smol,
                                                             model='PCM',
                                                             seed=546)

        self.syn_data_larg = create_synthetic_irt_polytomous(self.difficulty,
                                                             self.discrimination,
                                                             thetas,
                                                             model='PCM',
                                                             seed=546)

        self.syn_data_mixed = create_synthetic_irt_polytomous(self.difficulty_smol[:, 1:],
                                                              self.discrimination_smol,
                                                              thetas,
                                                              model='PCm',
                                                              seed=543)

    def test_pcm_gets_better(self):
        """Testing mml partial credit model improves with parameters."""
        output_smol = pcm_mml(self.syn_data_smol)
        output_large = pcm_mml(self.syn_data_larg)

        def rmse(expected, result):
            return np.sqrt(np.nanmean(np.square(expected - result)))

        rmse_smol = rmse(output_smol['Discrimination'], self.discrimination_smol)
        rmse_large = rmse(output_large['Discrimination'][:5], self.discrimination_smol)

        self.assertLess(rmse_large, rmse_smol)
        self.assertAlmostEqual(rmse_large, 0.0728439345, places=4)

        # Regression Tests
        expected_discr = np.array([0.81237, 0.98594, 1.15784, 
                                   0.54352, 1.07744, 0.53615, 1.04752,
                                   0.82479, 1.13313, 0.52348])
        expected_diff = np.array(
                            [[-0.36228655,  1.12552716, -0.8683124,  -0.05670265],
                            [-0.74530574,  0.50813067, -1.3035864,   1.54851965],
                            [-0.37363133,  0.22108749,  1.49891726,  0.82861256],
                            [ 0.61968191, -1.62910971, -0.1741578,  -0.64065034],
                            [-0.7778711,  -1.53569334,  0.20636619,  0.74253886],
                            [ 0.34621761, -1.9338782,  1.02446975, -0.60102526],
                            [-1.05718863, -0.99201518,  0.78186304,  1.02658717],
                            [ 0.34289683,  0.60301358,  1.80654666, -1.16526188],
                            [ 0.83154823, -2.08790263,  0.72526695, -1.12897981],
                            [-0.22679954, -1.18577072,  1.94080517, -0.45182054]])

        np.testing.assert_allclose(
            expected_discr, output_large['Discrimination'], atol=1e-3, rtol=1e-3)
        np.testing.assert_allclose(
            expected_diff, output_large['Difficulty'], atol=1e-3, rtol=1e-3)

    def test_pcm_mixed_difficulty_length(self):
        """Testing response set with different difficulty lengths."""
        syn_data = self.syn_data_larg.copy()
        syn_data[:5, :] = self.syn_data_mixed

        expected_diff = np.array([[ 1.41437467, -1.03972326, -0.03385356,      np.nan],
                                  [ 0.39334941, -1.14674155,  1.43811448,      np.nan],
                                  [ 0.13997558,  1.48476978,  0.91178314,      np.nan],
                                  [-2.15131714, -0.38478815, -0.18543023,      np.nan],
                                  [-1.41427106,  0.0185614,   0.74579592,      np.nan],
                                  [ 0.44808276, -2.07073615,  1.08686984, -0.71594868],
                                  [-1.05742444, -1.02596807,  0.78562506,  1.02584723],
                                  [ 0.28810824,  0.5646581,   1.74534348, -1.05365121],
                                  [ 0.72990561, -2.03026853,  0.65491416, -1.05760536],
                                  [-0.2213386,  -1.21817327,  1.96700887, -0.48493213]])

        expected_discr = np.array([0.67902894, 1.20353141, 1.20042395, 0.52139593, 1.30484326,
                                   0.49529043, 1.01706525, 0.86600196, 1.19555753, 0.51229351])

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
        np.random.seed(555)
        difficulty = -(np.random.rand(10, 2) * 1.5)
        delta = 0.5 * np.random.rand(10, 1)
        discrimination = np.random.rand(10) + 0.5
        thetas = np.random.randn(200)

        betas = np.c_[difficulty, np.zeros((10, 1)), -difficulty[:, ::-1]]
        betas += delta
        syn_data = create_synthetic_irt_polytomous(betas, discrimination,
                                                   thetas, model='GUM',
                                                   seed=546)

        result = gum_mml(syn_data, {'max_iteration': 100})

        rmse_discrimination = np.sqrt(np.square(discrimination.squeeze() - 
                                                result['Discrimination']).mean())
        rmse_delta = np.sqrt(np.square(delta.squeeze() - 
                                       result['Delta']).mean())
        rmse_tau = np.sqrt(np.square(difficulty - 
                                     result['Tau']).mean())

        self.assertAlmostEqual(rmse_discrimination, 0.371009140, places=4)
        self.assertAlmostEqual(rmse_delta, 0.893843069403, places=4)
        self.assertAlmostEqual(rmse_tau, 0.6406162162, places=4)


if __name__ == '__main__':
    unittest.main()
