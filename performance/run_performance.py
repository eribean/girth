import json
import argparse
from functools import partial

import numpy as np
import scipy as sp

import girth
from girth.performance import (validate_performance_dict,
                               scipy_stats_string_to_functions)

parser_description = """Runs simulations of item response models given a
                        configuration file.  For more information see
                        documentation at

                              https://eribean.github.io/girth/"""


def create_item_parameters(analysis_dict):
    """ Creates the synthetic data to run metrics on.

        Args:
            analysis_dict:  Dictionary of item parameters

        Returns:
            Iterator that yields a synthetic dataset 
    """
    np.random.seed(analysis_dict['Seed'])

    # Create the discrimination parameters
    if analysis_dict['Discrimination_fixed'] is not None:
        discrimination = np.atleast_1d(analysis_dict['Discrimination_fixed'])
    
    else:
        pdf = scipy_stats_string_to_functions(analysis_dict['Discrimination_pdf'],
                                              analysis_dict['Discrimination_pdf_args'])
        discrimination = pdf.rvs(size=analysis_dict['Discrimination_count'])

    # Create the difficulty parameters
    if analysis_dict['Difficulty_fixed'] is not None:
        difficulty = np.atleast_1d(analysis_dict['Difficulty_fixed'])
    
    else:
        pdf = scipy_stats_string_to_functions(analysis_dict['Difficulty_pdf'],
                                              analysis_dict['Difficulty_pdf_args'])
        difficulty = pdf.rvs(size=analysis_dict['Difficulty_count'])

    if analysis_dict['Type'].lower() == "polytomous":
        poly_type = {'graded': 'grm', 'credit': 'pcm', 
                     'unfold': 'gum'}[analysis_dict['SubType'].lower()]
        
        if poly_type == 'grm':
            # Only allow ordered
            difficulty = np.sort(difficulty, axis=1)
        
        elif poly_type == 'gum':
            # must be skew symmetric about middle
            middle_index = (difficulty.shape[1] - 1) // 2
            offset = difficulty[:, middle_index].copy()
            difficulty -= offset[:, None]
            difficulty[:, :middle_index] = -difficulty[:, middle_index+1:][:, ::-1]
            difficulty += offset[:, None]
    
    return discrimination, difficulty


def create_synthetic_data(difficulty, discrimination, analysis_dict, ndx):
    """ Creates the synthetic data to run metrics on.

        Args:
            analysis_dict:  Dictionary of item parameters

        Returns:
            Iterator that yields a synthetic dataset 
    """
    seed = analysis_dict['Seed'] + 1 + ndx

    ability = scipy_stats_string_to_functions(analysis_dict['Ability_pdf'],
                                              analysis_dict['Ability_pdf_args'])
    ability_counts = analysis_dict['Ability_count'][ndx]

    if analysis_dict['Type'].lower() == "dichotomous":
        func = partial(girth.create_synthetic_irt_dichotomous,
                       difficulty=difficulty, discrimination=discrimination)
    else:
        poly_type = {'graded': 'grm', 'credit': 'pcm', 
                     'unfold': 'gum'}[analysis_dict['SubType'].lower()]
        
        func = partial(girth.create_synthetic_irt_polytomous,
                       difficulty=difficulty, discrimination=discrimination,
                       model=poly_type) 

    for _ in range(analysis_dict['Trials']):
        seed += 1
        thetas = ability.rvs(size=ability_counts)
        yield func(thetas=thetas, seed=seed)


def gather_metrics(synthesis, analysis, ndx):
    """ Runs the synthetic data creation
 
        Args:
            synthesis: Dictionary of synthesis parameters
            analysis: Dictionary of analysis parameters
        
        Return metrics 
    """
    alpha, beta = create_item_parameters(synthesis)

    datasets = create_synthetic_data(beta, alpha, synthesis, 0)

    estimate_func = girth.__dict__[analysis['Model']]

    rmse_alpha = list()
    rmse_beta = list()

    for dataset in datasets:
        otpt = estimate_func(dataset)

        value = np.sqrt(np.nanmean(np.square(otpt[0] - alpha)))
        rmse_alpha.append(value)

        value = np.sqrt(np.nanmean(np.square(otpt[1] - beta)))
        rmse_beta.append(value)
    
    return (rmse_alpha, rmse_beta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="GIRTH Performance Script",
                                     description=parser_description)
    parser.add_argument("-c", "--config", required=True, 
                        help="JSON file of script parameters")
    parser.add_argument("-o", "--output_dir", default="./", 
                        help="Output Directory to save results")

    # Read in the arguments
    args = parser.parse_args()

    with open(args.config, 'r') as fptr:
        config_dict = json.load(fptr)
    
    config_dict = validate_performance_dict(config_dict)

    a, b = gather_metrics(config_dict['Analysis'], 
                          config_dict['Synthesis']["Rasch_Approx"], 0)

    print(np.mean(a), np.std(a, ddof=1))
    print(np.mean(b), np.std(b, ddof=1))