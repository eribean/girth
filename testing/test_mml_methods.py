import unittest

import numpy as np

from girth import create_synthetic_irt_dichotomous
from girth import rasch_approx, onepl_approx, twopl_approx
from girth import rasch_separate, onepl_separate, twopl_separate
from girth import rasch_full, onepl_full, twopl_full

from girth import create_synthetic_irt_polytomous
from girth import grm_separate, pcm_full


class TestMMLRaschMethods(unittest.TestCase):

    ### REGRESSION TESTS

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


    def test_rasch_regression_approximate(self):
        """Testing rasch approximation methods."""
        syn_data = self.data.copy()
        output = rasch_approx(syn_data, self.discrimination)
        expected_output = np.array([-1.27477108, -0.7771253 , -0.07800756,
                                    0.62717748,  1.41945661])

        np.testing.assert_allclose(expected_output, output)


    def test_rasch_regression_separate(self):
        """Testing rasch separate methods."""
        syn_data = self.data.copy()
        output = rasch_separate(syn_data, self.discrimination)
        expected_output = np.array([-1.32474665, -0.81460991, -0.08221992,
                                     0.65867573,  1.47055368])

        np.testing.assert_allclose(expected_output, output)


    def test_rasch_regression_full(self):
        """Testing rasch full methods."""
        syn_data = self.data.copy()
        output = rasch_full(syn_data, self.discrimination)
        expected_output = np.array([-1.3221573 , -0.81445556, -0.08485538,
                                    0.65457445,  1.4664268])

        np.testing.assert_allclose(expected_output, output)

    def test_rasch_close(self):
        """Testing rasch converging methods."""
        np.random.seed(333)
        difficulty = np.linspace(-1.25, 1.25, 5)
        discrimination = 0.87
        thetas = np.random.randn(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = rasch_separate(syn_data, discrimination)
        np.testing.assert_array_almost_equal(difficulty, output, decimal=1)


class TestMMLOnePLMethods(unittest.TestCase):

    ### REGRESSION TESTS

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


    def test_onepl_regression_approximate(self):
        """Testing onepl approximation methods."""
        syn_data = self.data.copy()
        output = onepl_approx(syn_data)
        expected_output = np.array([-1.31442222, -0.6020974 , -0.03617573,
                                    0.72561239,  1.32432846])

        self.assertAlmostEqual(output[0], 1.9339261148822318)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_onepl_regression_separate(self):
        """Testing onepl separate methods."""
        syn_data = self.data.copy()
        output = onepl_separate(syn_data)
        expected_output = np.array([-1.37650768, -0.64900385, -0.0393339 ,
                                    0.7791904 ,  1.38618721])

        self.assertAlmostEqual(output[0], 1.901703384)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_onepl_regression_full(self):
        """Testing onepl full methods."""
        syn_data = self.data.copy()
        output = onepl_full(syn_data)
        expected_output = np.array([-1.37891489, -0.64731397, -0.03576614,
                                     0.78093483,  1.38451727])

        self.assertAlmostEqual(output[0], 1.9017531986)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_oneple_close(self):
        """Testing rasch converging methods."""
        np.random.seed(843)
        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.87
        thetas = np.random.randn(1000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = onepl_separate(syn_data)
        self.assertLess(np.abs(output[0] - discrimination).max(), 0.1)
        self.assertLess(np.abs(output[1] - difficulty).max(), 0.2)


class TestMMLTwoPLMethods(unittest.TestCase):

    ### REGRESSION TESTS

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


    def test_twopl_regression_approximate(self):
        """Testing onepl approximation methods."""
        syn_data = self.data.copy()
        output = twopl_approx(syn_data)

        expected_discrimination = np.array([0.99992967, 1.86001065, 1.36698371, 
                                            0.53176185, 0.90445937])
        expected_output = np.array([-1.26863512, -0.60677766, -0.07459365,  
                                    0.75672082,  1.62766257])

        np.testing.assert_allclose(expected_discrimination, output[0], rtol=1e-6)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_twopl_regression_separate(self):
        """Testing onepl separate methods."""
        syn_data = self.data.copy()
        output = twopl_separate(syn_data)

        expected_discrimination = np.array([0.99973958, 1.86359282, 1.35543477, 
                                            0.52934393, 0.90911332])
        expected_output = np.array([-1.31515501, -0.64828062, -0.07980151,  
                                     0.77407585,  1.66774338])

        np.testing.assert_allclose(expected_discrimination, output[0], rtol=1e-6)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_twopl_regression_full(self):
        """Testing onepl full methods."""
        syn_data = self.data.copy()
        output = twopl_full(syn_data)

        expected_discrimination = np.array([0.99976507, 1.86290853, 1.35540523, 
                                            0.5293102 , 0.91224903])
        expected_output = np.array([-1.31492755, -0.64812777, -0.08017468,  
                                     0.77399901,  1.66321445])

        np.testing.assert_allclose(expected_discrimination, output[0], rtol=1e-6)
        np.testing.assert_allclose(expected_output, output[1], rtol=1e-6)


    def test_twopl_close(self):
        """Testing rasch converging methods."""
        np.random.seed(43)
        difficulty = np.linspace(-1.25, 1.25, 10)
        discrimination = 0.5 + np.random.rand(10)
        thetas = np.random.randn(2000)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination,
                                                    thetas)

        output = twopl_separate(syn_data)
        self.assertLess(np.abs(output[0] - discrimination).mean(), 0.1)
        self.assertLess(np.abs(output[1] - difficulty).mean(), 0.1)        


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

        estimated_parameters = grm_separate(syn_data)

        # Regression test
        expected_discrimination = np.array([0.58291387, 1.38349382, 0.87993092, 
                                            1.17329774, 1.47195824])

        expectected_difficulty = np.array([[-1.23881672, -0.5340817 ,  0.06170343,  1.23881201],
                                           [-1.09279868, -0.76747967,  0.44660955,  1.28909032],
                                           [-0.16828803,  0.23943693,  0.76140209,  1.24435541],
                                           [-2.02935022, -0.70214267, -0.23281603,  1.27521818],
                                           [-1.47758497, -0.9050062 ,  0.0698804 ,  0.71286592]])

        np.testing.assert_array_almost_equal(estimated_parameters[0], expected_discrimination)
        np.testing.assert_array_almost_equal(estimated_parameters[1], expectected_difficulty)
        

    def test_graded_small_participant(self):
        """Regression Testing graded response model with small N."""
        np.random.seed(87)
        difficulty = np.sort(np.random.randn(5, 4), axis=1)
        discrimination = np.random.rand(5) + 0.5
        thetas = np.random.randn(101)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        estimated_parameters = grm_separate(syn_data)

        # Regression test
        expected_discrimination = np.array([1.72485717, 0.39305266, 0.82841429, 
                                            0.93731447, 1.56774651])

        expectected_difficulty = np.array([[-0.1571751 ,  0.22757599,  0.33438484,  0.59311136],
                                           [ 1.22266539,  1.3332354 ,  2.26915694,  3.51488207],
                                           [-1.25515215, -1.06235363, -0.70073232,  0.87821931],
                                           [-1.26630059, -0.37880122,  2.25720247,      np.nan],
                                           [-1.36961213, -0.66186681, -0.50308306,      np.nan]])

        np.testing.assert_array_almost_equal(estimated_parameters[0], expected_discrimination)
        np.testing.assert_array_almost_equal(estimated_parameters[1], expectected_difficulty)
        

    def test_graded_response_model_close(self):
        """Regression Testing graded response model with large N."""
        np.random.seed(6322)
        difficulty = np.sort(np.random.randn(10, 4), axis=1)
        discrimination = np.random.rand(10) + 0.5
        thetas = np.random.randn(1000)
        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination,
                                                   thetas)

        estimated_parameters = grm_separate(syn_data)

        rmse = np.sqrt(np.square(estimated_parameters[0] - discrimination).mean())
        self.assertLess(rmse, .0966)

        rmse = np.sqrt(np.square(estimated_parameters[1] - difficulty).mean())
        self.assertLess(rmse, .1591)

        
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
        output_smol = pcm_full(self.syn_data_smol)
        output_large = pcm_full(self.syn_data_larg)

        def rmse(expected, result):
            return np.sqrt(np.nanmean(np.square(expected - result)))
        
        rmse_smol = rmse(output_smol[0], self.discrimination_smol)
        rmse_large = rmse(output_large[0][:5], self.discrimination_smol)

        self.assertLess(rmse_large, rmse_smol)
        self.assertAlmostEqual(rmse_large, 0.07285609)

        # Regression Tests
        expected_discr = np.array([0.81233975, 0.98594227, 1.15784476, 
                                   0.54351843, 1.0774421, 0.53615107, 
                                   1.0475184 , 0.82479055, 1.13312411, 
                                   0.52347491])

        expected_diff = np.array([[-0.36227744,  1.12566146, -0.86842382, -0.05673954],
                                  [-0.74530762,  0.50813106, -1.30358698,  1.5485216 ],
                                  [-0.37363539,  0.22108959,  1.49892019,  0.82861686],
                                  [ 0.61968408, -1.62910758, -0.17415797, -0.64065788],
                                  [-0.7778677 , -1.53569663,  0.2063686 ,  0.74254058],
                                  [ 0.34622206, -1.93388653,  1.02447532, -0.60102884],
                                  [-1.05718775, -0.99201864,  0.78186396,  1.02659318],
                                  [ 0.34289862,  0.6030145 ,  1.80655127, -1.16526299],
                                  [ 0.83155461, -2.08791222,  0.72527178, -1.12898438],
                                  [-0.22679832, -1.18577464,  1.94080754, -0.45182257]])

        np.testing.assert_array_almost_equal(expected_discr, output_large[0], decimal=5)
        np.testing.assert_array_almost_equal(expected_diff, output_large[1], decimal=5)


    def test_pcm_mixed_difficulty_length(self):
        """Testing response set with different difficulty lengths."""
        syn_data = self.syn_data_larg.copy()
        syn_data[:5, :] = self.syn_data_mixed

        expected_diff = np.array([[ 1.41437983, -1.03972953, -0.03385635,      np.nan],
                                  [ 0.39319676, -1.14680525,  1.43869585,      np.nan],
                                  [ 0.13998335,  1.48481773,  0.91181363,      np.nan],
                                  [-2.15135695, -0.38479504, -0.18542855,      np.nan],
                                  [-1.41430646,  0.01855994,  0.74583287,      np.nan],
                                  [ 0.44810092, -2.07080129,  1.08690341, -0.71594963],
                                  [-1.05746582, -1.02600005,  0.78565529,  1.02590547],
                                  [ 0.28814764,  0.56462721,  1.74554608, -1.05376351],
                                  [ 0.72998838, -2.03041128,  0.65498806, -1.05767631],
                                  [-0.22134409, -1.21822046,  1.96707249, -0.48492872]])

        expected_discr = np.array([0.67902721, 1.20333533, 1.20038813, 0.52138263, 
                                   1.30479874, 0.49527629, 1.01702145, 0.86596228, 
                                   1.1954911 , 0.51227604])

        output = pcm_full(syn_data)
        np.testing.assert_array_almost_equal(expected_diff, output[1], decimal=5)
        np.testing.assert_array_almost_equal(expected_discr, output[0], decimal=5)




if __name__ == '__main__':
    unittest.main()
