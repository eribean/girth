#pylint: skip-file
import json
import argparse
import multiprocessing
from functools import partial
from itertools import product, starmap, repeat
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
    rng = np.random.default_rng(analysis_dict['Seed'])

    # Create the discrimination parameters
    if analysis_dict['Discrimination_fixed'] is not None:
        discrimination = np.atleast_1d(analysis_dict['Discrimination_fixed'])
    
    else:
        pdf = scipy_stats_string_to_functions(analysis_dict['Discrimination_pdf'],
                                              analysis_dict['Discrimination_pdf_args'])
        discrimination = pdf.rvs(size=analysis_dict['Discrimination_count'], 
                                 random_state=rng)

    # Create the difficulty parameters
    if analysis_dict['Difficulty_fixed'] is not None:
        difficulty = np.atleast_1d(analysis_dict['Difficulty_fixed'])
    
    else:
        pdf = scipy_stats_string_to_functions(analysis_dict['Difficulty_pdf'],
                                              analysis_dict['Difficulty_pdf_args'])
        difficulty = pdf.rvs(size=analysis_dict['Difficulty_count'],
                             random_state=rng)

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


def create_synthetic_data(difficulty, discrimination, analysis_dict, 
                          ability_ndx, trial_ndx):
    """ Creates the synthetic data to run metrics on.

        Args:
            analysis_dict:  Dictionary of item parameters

        Returns:
            Iterator that yields a synthetic dataset 
    """
    seed = analysis_dict['Seed'] + 1 + ability_ndx

    ability = scipy_stats_string_to_functions(analysis_dict['Ability_pdf'],
                                              analysis_dict['Ability_pdf_args'])
    ability_counts = analysis_dict['Ability_count'][ability_ndx]

    if analysis_dict['Type'].lower() == "dichotomous":
        func = partial(girth.create_synthetic_irt_dichotomous,
                       difficulty=difficulty, discrimination=discrimination)
    else:
        poly_type = {'graded': 'grm', 'credit': 'pcm', 
                     'unfold': 'gum'}[analysis_dict['SubType'].lower()]
        
        func = partial(girth.create_synthetic_irt_polytomous,
                       difficulty=difficulty, discrimination=discrimination,
                       model=poly_type)
    
    # Fast forward the random seed to make sure
    # results are replicable
    seed += trial_ndx[0]

    for _ in range(trial_ndx[0], trial_ndx[1]):
        seed += 1
        thetas = ability.rvs(size=ability_counts, random_state=2*seed)
        yield func(thetas=thetas, seed=seed)


def gather_metrics(synthesis_key, ability_ndx, trial_ndx, config_dict):
    """ Runs the synthetic data creation
 
        Args:
            synthesis_key: Dictionary key into synthesis parameters
            ability_ndx: Index into ability count
            trial_ndx: integer into ability count
            config_dict: dictionary of run parameters
        
        Returns:
            dictionary of metrics for run for the trial nds
    """
    analysis = config_dict['Analysis']
    synthesis = config_dict['Synthesis'][synthesis_key]

    alpha, beta = create_item_parameters(analysis)

    datasets = create_synthetic_data(beta, alpha, analysis,
                                     ability_ndx, trial_ndx)

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
    count = analysis['Ability_count'][ability_ndx]
    return {"key": synthesis_key,
            "count": count,
            "data": {'alpha': rmse_alpha,
                     'beta': rmse_beta,
                     'time': time_counts}}


def compute_statistics(synthesis_key, ability_count, results_dict):
    """Filters the results and recombines, then takes statistics."""
    def _statistics(input_array):
        return {'max': np.max(input_array),
                'min': np.min(input_array),
                'mean': np.mean(input_array),
                'std': np.std(input_array, ddof=1),
                'quartile_1': np.percentile(input_array, 25),
                'quartile_2': np.percentile(input_array, 75)}

    dataset = list(filter(lambda x: (x['key'] == synthesis_key) & (x['count'] == ability_count), 
                          results_dict))
    
    rmse_alpha = np.concatenate([x['data']['alpha'] for x in dataset])
    rmse_beta = np.concatenate([x['data']['beta'] for x in dataset])
    time_counts = np.concatenate([x['data']['time'] for x in dataset])

    return {ability_count: {'alpha': _statistics(rmse_alpha),
                            'beta': _statistics(rmse_beta),
                            'time': _statistics(time_counts)}} 


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
    t1 = time()
    # Make sure values are working
    config_dict = validate_performance_dict(config_dict)

    # get the appropriate map function
    processor_count = config_dict['Options']['Processor_count']
    map_func = starmap
    if processor_count > 1:
        map_func = multiprocessing.Pool(processes=processor_count).starmap

    # Create the interation indices for the chunks
    number_of_trials = (1 + config_dict['Analysis']['Trials'] // 
                        config_dict['Options']['Chunksize'])

    trial_chunks = np.linspace(0, config_dict['Analysis']['Trials'], 
                               number_of_trials, dtype=int)
    trial_chunks = list(zip(trial_chunks[:-1], trial_chunks[1:]))
    synthesis_keys = config_dict['Synthesis'].keys()

    chunks = product(synthesis_keys, range(len(config_dict['Analysis']['Ability_count'])), 
                     trial_chunks)

    print(f"Using {config_dict['Options']['Processor_count']} processors\n")
    print(f"Processing {len(synthesis_keys)*len(config_dict['Analysis']['Ability_count']) * len(trial_chunks)} chunks")

    # Process the outputs
    function_call = partial(gather_metrics, config_dict=config_dict)
    results = map_func(function_call, chunks)    
    results = list(results)

    # Combine statistics for all the data
    results_dict = dict()
    for synthesis_key in synthesis_keys:
        results_dict[synthesis_key] = dict()
        for ability_count in config_dict['Analysis']['Ability_count']:
            results_stats = compute_statistics(synthesis_key, ability_count, results)
            results_dict[synthesis_key].update(results_stats)

    # Add the truth datasets
    alpha, beta = create_item_parameters(config_dict['Analysis'])
    results_dict['Truth'] = {'alpha': alpha.tolist(),
                             'beta': beta.tolist()}   
    # write out file
    output_file = path.join(args.output_dir, 
                            config_dict['Name'].replace(' ', '_') + '.json')
    
    with open(output_file, 'w') as fptr:
        json.dump(results_dict, fptr, indent=4)
    
    print(f"Total processing time = {time() - t1}")