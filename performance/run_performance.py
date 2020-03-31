import json
import argparse
import multiprocessing
from functools import partial
from time import time
from os import path

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


def gather_metrics(analysis, synthesis, ndx):
    """ Runs the synthetic data creation
 
        Args:
            analysis: Dictionary of analysis parameters
            synthesis: Dictionary of synthesis parameters
            ndx: integer into ability count
        
        Returns:
            dictionary of metrics for run 
    """
    alpha, beta = create_item_parameters(analysis)

    datasets = create_synthetic_data(beta, alpha, analysis, ndx)

    estimate_func = girth.__dict__[synthesis['Model']]

    rmse_alpha = list()
    rmse_beta = list()
    time_counts = list()

    for dataset in datasets:
        t1 = time()
        otpt = estimate_func(dataset)
        time_counts.append(time() - t1)

        if type(otpt) is tuple:
            value = np.sqrt(np.nanmean(np.square(otpt[0] - alpha)))
            rmse_alpha.append(value)

            value = np.sqrt(np.nanmean(np.square(otpt[1] - beta)))
            rmse_beta.append(value)
        
        else:
            rmse_alpha.append(np.abs(alpha))
            value = np.sqrt(np.nanmean(np.square(otpt[0] - beta)))
            rmse_beta.append(value)           
    
    # Package outputs
    count = analysis['Ability_count'][ndx]
    def _statistics(input_array):
        return {'max': np.max(input_array),
                'min': np.min(input_array),
                'mean': np.mean(input_array),
                'std': np.std(input_array, ddof=1),
                'quartile_1': np.percentile(input_array, 25),
                'quartile_2': np.percentile(input_array, 75)}

    output = {count: {'alpha': _statistics(rmse_alpha),
                      'beta': _statistics(rmse_beta),
                      'time': _statistics(time_counts)}}    
    return output


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
    
    # Make sure values are working
    config_dict = validate_performance_dict(config_dict)

    # get the appropriate map function
    processor_count = config_dict['Options']['Processor_count']
    map_func = map
    if processor_count > 1:
        map_func = multiprocessing.Pool(processes=processor_count).map

    # sort items from largest to smallest
    ability_counts = config_dict["Analysis"]["Ability_count"]
    ability_counts.sort(reverse=True)
    ability_length = len(ability_counts)

    print(f"Using {config_dict['Options']['Processor_count']} processors\n")

    results_dict = dict()
    # Loop over synthesis arguments
    for key, synthesis in config_dict['Synthesis'].items():
        print(f"Start processing on {key}")
        results_dict[key] = {}
        
        # This function will be evaluted at each ability count
        run_func = partial(gather_metrics, config_dict['Analysis'],
                           synthesis)
    
        results = map_func(run_func, range(ability_length))
    
        for ndx, result in enumerate(results):
            results_dict[key].update(result)
    
    alpha, beta = create_item_parameters(config_dict['Analysis'])
    results_dict['Truth'] = {'alpha': alpha.tolist(),
                             'beta': beta.tolist()}   
    # write out file
    output_file = path.join(args.output_dir, 
                            config_dict['Name'].replace(' ', '_') + '.json')
    
    with open(output_file, 'w') as fptr:
        json.dump(results_dict, fptr, indent=4)