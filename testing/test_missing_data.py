import unittest  # pylint: disable=cyclic-import

import numpy as np

from girth import tag_missing_data

from girth import create_synthetic_irt_dichotomous, create_synthetic_irt_polytomous
from girth import rasch_jml, onepl_jml, twopl_jml
from girth import rasch_mml, onepl_mml, twopl_mml
from girth import rasch_full, onepl_full, twopl_full

from girth import grm_jml, pcm_jml
from girth import grm_mml_eap, grm_mml, pcm_mml, gum_mml


def _create_missing_data(dataset, seed, threshold):
    """Creates missing data for unittests."""
    np.random.seed(seed)
    mask = np.random.rand(*dataset.shape) < threshold
    dataset2 = dataset.copy()
    dataset2[mask] = 2222
    return dataset2


def _rmse(dataset1, dataset2):
    return np.sqrt(np.square(dataset1 - dataset2).mean())


class TestMissingDichotomous(unittest.TestCase):
    """Testing Dichotomous methods for missing data."""

    ## SMOKE / REGRESSION TESTS with human-in-loop spot check

    def test_rasch_jml_mml(self):
        """Testing rasch mml/jml for missing data."""
        np.random.seed(7321)
        n_items = 10
        n_people = 200

        difficulty = np.random.randn(n_items)
        theta = np.random.randn(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, 1.0, theta)
        syn_data_tagged = _create_missing_data(syn_data, 73435, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [0, 1])

        # JML
        result_all_good = rasch_jml(syn_data)
        result_missing = rasch_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.072384, 4)

        # MML
        result_all_good = rasch_mml(syn_data)
        result_missing = rasch_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.054726, 4)

        # MML FULL
        result_all_good = rasch_full(syn_data)
        result_missing = rasch_full(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.0590626, 4)        

    def test_onepl_jml_mml(self):
        """Testing oneple mml/jml for missing data."""
        np.random.seed(789595)
        n_items = 10
        n_people = 200

        discrimination = 1.53
        difficulty = np.random.randn(n_items)
        theta = np.random.randn(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, theta)
        syn_data_tagged = _create_missing_data(syn_data, 66877, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [0, 1])

        # JML 
        result_all_good = onepl_jml(syn_data)
        result_missing = onepl_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.048643, 4)        
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.017526, 4)        

        # MML
        result_all_good = onepl_mml(syn_data)
        result_missing = onepl_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.031971, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.045726, 4)

        # MML FULL
        result_all_good = onepl_full(syn_data)
        result_missing = onepl_full(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.038550, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.052542, 4)

    def test_twopl_jml_mml(self):
        """Testing twopl mml/jml for missing data."""
        np.random.seed(15335)
        n_items = 10
        n_people = 200

        discrimination = np.sqrt(-2 * np.log(np.random.rand(n_items)))
        difficulty = np.random.randn(n_items)
        theta = np.random.randn(n_people)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, theta)
        syn_data_tagged = _create_missing_data(syn_data, 84433, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [0, 1])

        # JML 
        result_all_good = twopl_jml(syn_data)
        result_missing = twopl_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.283604, 4)        
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.669763, 4)        

        # MML
        result_all_good = twopl_mml(syn_data)
        result_missing = twopl_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.064496, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.155868, 4)

        # MML FULL
        result_all_good = twopl_full(syn_data)
        result_missing = twopl_full(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.066202, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.166226, 4)


class TestMissingPolytomous(unittest.TestCase):
    """Testing Polytomous methods for missing data."""    

    def test_grm_jml_mml(self):
        """Testing graded response mml/jml for missing data."""
        np.random.seed(69871)
        n_items = 10
        n_people = 300

        discrimination = np.sqrt(-2 * np.log(np.random.rand(n_items)))
        difficulty = np.random.randn(n_items, 3)
        difficulty = np.sort(difficulty, 1)
        theta = np.random.randn(n_people)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='grm')
        syn_data_tagged = _create_missing_data(syn_data, 879858, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [1, 2, 3, 4])

        # JML
        result_all_good = grm_jml(syn_data)
        result_missing = grm_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.097251, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.112861, 4)

        # MML
        result_all_good = grm_mml(syn_data)
        result_missing = grm_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.088626, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.065834, 4)

        # EAP/MML
        result_all_good = grm_mml_eap(syn_data)
        result_missing = grm_mml_eap(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.073556, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.062288, 4)


    def test_pcm_jml_mml(self):
        """Testing partial response mml/jml for missing data."""
        np.random.seed(499867)
        n_items = 10
        n_people = 300

        discrimination = np.sqrt(-2 * np.log(np.random.rand(n_items)))
        difficulty = np.random.randn(n_items, 3)
        difficulty = np.sort(difficulty, 1)
        theta = np.random.randn(n_people)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='pcm')
        syn_data_tagged = _create_missing_data(syn_data, 1986778, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [1, 2, 3, 4])

        # JML
        result_all_good = pcm_jml(syn_data)
        result_missing = pcm_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.078015, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.147213, 4)
        
        # MML
        result_all_good = pcm_mml(syn_data)
        result_missing = pcm_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.085013, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.079640, 4)


    def test_gum_mml(self):
        """Testing unfolding response mml/jml for missing data."""
        np.random.seed(7382)
        n_items = 15
        n_people = 600

        discrimination = np.sqrt(-2 * np.log(np.random.rand(n_items)))
        delta = np.random.randn(n_items, 1)
        difficulty = np.random.randn(n_items, 3)
        difficulty = np.sort(difficulty, 1)
        difficulty = np.c_[-difficulty, np.zeros((n_items, 1)), difficulty[:, ::-1]] + delta
        theta = np.random.randn(n_people)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='gum')
        syn_data_tagged = _create_missing_data(syn_data, 112, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [1, 2, 3, 4])

        # MML
        result_all_good = gum_mml(syn_data)
        result_missing = gum_mml(syn_data_missing)

        difference_rmse = _rmse(result_all_good['Delta'], result_missing['Delta'])
        self.assertAlmostEqual(difference_rmse, 0.0664, 3)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.1382, 2)

if __name__ == '__main__':
    unittest.main()
