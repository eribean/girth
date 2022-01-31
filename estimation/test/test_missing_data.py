import unittest  # pylint: disable=cyclic-import

import numpy as np

from girth.synthetic import (create_synthetic_irt_dichotomous, 
    create_synthetic_irt_polytomous)
from girth import (tag_missing_data, rasch_jml, onepl_jml, twopl_jml,
    rasch_mml, onepl_mml, twopl_mml, grm_jml, pcm_jml,
    grm_mml_eap, grm_mml, pcm_mml, gum_mml)


def _create_missing_data(dataset, rng, threshold):
    """Creates missing data for unittests."""
    mask = rng.uniform(0, 1, dataset.shape) < threshold
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
        rng = np.random.default_rng(16241891)
        n_items = 10
        n_people = 200

        difficulty = rng.standard_normal(n_items)
        theta = rng.standard_normal(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, 1.0, theta, seed=rng)
        syn_data_tagged = _create_missing_data(syn_data, rng, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [0, 1])

        # JML
        result_all_good = rasch_jml(syn_data)
        result_missing = rasch_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.037436338, 4)

        # MML
        result_all_good = rasch_mml(syn_data)
        result_missing = rasch_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.034303, 4)      

    def test_onepl_jml_mml(self):
        """Testing oneple mml/jml for missing data."""
        rng = np.random.default_rng(778445138)
        n_items = 10
        n_people = 200

        discrimination = 1.53
        difficulty = rng.standard_normal(n_items)
        theta = rng.standard_normal(n_people)
        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, theta, seed=rng)
        syn_data_tagged = _create_missing_data(syn_data, rng, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [0, 1])

        # JML 
        result_all_good = onepl_jml(syn_data)
        result_missing = onepl_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.1173865, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.0979472, 4)        

        # MML
        result_all_good = onepl_mml(syn_data)
        result_missing = onepl_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.0456441, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.0011834, 4)

    def test_twopl_jml_mml(self):
        """Testing twopl mml/jml for missing data."""
        rng = np.random.default_rng(9845132189)
        n_items = 10
        n_people = 200

        discrimination = np.sqrt(-2 * np.log(rng.uniform(0, 1, n_items)))
        difficulty = rng.standard_normal(n_items)
        theta = rng.standard_normal(n_people)

        syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, theta, seed=rng)
        syn_data_tagged = _create_missing_data(syn_data, rng, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [0, 1])

        # JML 
        result_all_good = twopl_jml(syn_data)
        result_missing = twopl_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.253499856, 4)        
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.3829233, 4)        

        # MML
        result_all_good = twopl_mml(syn_data, {'use_LUT': False})
        result_missing = twopl_mml(syn_data_missing, {'use_LUT': False})
        difference_rmse = _rmse(result_all_good['Difficulty'], result_missing['Difficulty'])
        self.assertAlmostEqual(difference_rmse, 0.12724734, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.225398, 4)


class TestMissingPolytomous(unittest.TestCase):
    """Testing Polytomous methods for missing data."""    

    def test_grm_jml_mml(self):
        """Testing graded response mml/jml for missing data."""
        rng = np.random.default_rng(46842128)
        n_items = 10
        n_people = 300

        discrimination = np.sqrt(-2 * np.log(rng.uniform(0, 1, n_items)))
        difficulty = rng.standard_normal((n_items, 3))
        difficulty = np.sort(difficulty, 1)
        theta = rng.standard_normal(n_people)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='grm', seed=rng)
        syn_data_tagged = _create_missing_data(syn_data, rng, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [1, 2, 3, 4])

        # JML
        result_all_good = grm_jml(syn_data)
        result_missing = grm_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.17731254, delta=.002)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.290744, delta=.003)

        # MML
        result_all_good = grm_mml(syn_data)
        result_missing = grm_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.09786773, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.1786202, 4)

        # EAP/MML
        result_all_good = grm_mml_eap(syn_data)
        result_missing = grm_mml_eap(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.051064382, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.12787730, 4)

    def test_pcm_jml_mml(self):
        """Testing partial response mml/jml for missing data."""
        rng = np.random.default_rng(469871859211564)
        n_items = 10
        n_people = 300

        discrimination = np.sqrt(-2 * np.log(rng.uniform(0, 1, n_items)))
        difficulty = rng.standard_normal((n_items, 3))
        difficulty = np.sort(difficulty, 1)
        theta = rng.standard_normal(n_people)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='pcm', seed=rng)
        syn_data_tagged = _create_missing_data(syn_data, rng, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [1, 2, 3, 4])

        # JML
        result_all_good = pcm_jml(syn_data)
        result_missing = pcm_jml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.07731, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.073606, 4)
        
        # MML
        result_all_good = pcm_mml(syn_data)
        result_missing = pcm_mml(syn_data_missing)
        difference_rmse = _rmse(result_all_good['Difficulty'].ravel(), 
                                result_missing['Difficulty'].ravel())
        self.assertAlmostEqual(difference_rmse, 0.0758858, 4)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.06949731, 4)

    def test_gum_mml(self):
        """Testing unfolding response mml/jml for missing data."""
        rng = np.random.default_rng(68755852211)
        n_items = 15
        n_people = 600

        discrimination = np.sqrt(-2 * np.log(rng.uniform(0, 1, n_items)))
        delta = rng.standard_normal((n_items, 1))
        difficulty = rng.standard_normal((n_items, 3))
        difficulty = np.sort(difficulty, 1)
        difficulty = np.c_[-difficulty, np.zeros((n_items, 1)), difficulty[:, ::-1]] + delta
        theta = rng.standard_normal(n_people)

        syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                                   theta, model='gum', seed=rng)
        syn_data_tagged = _create_missing_data(syn_data, rng, 0.1)
        syn_data_missing = tag_missing_data(syn_data_tagged, [1, 2, 3, 4])

        # MML
        result_all_good = gum_mml(syn_data)
        result_missing = gum_mml(syn_data_missing)

        difference_rmse = _rmse(result_all_good['Delta'], result_missing['Delta'])
        self.assertAlmostEqual(difference_rmse, 0.0340275, 2)
        difference_rmse = _rmse(result_all_good['Discrimination'], result_missing['Discrimination'])
        self.assertAlmostEqual(difference_rmse, 0.0872884, 3)

if __name__ == '__main__':
    unittest.main()
