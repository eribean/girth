from multiprocessing import Pool

from itertools import starmap, repeat
from functools import partial

import numpy as np

from girth import validate_estimation_options


__all__ = ["standard_errors_bootstrap"]


def _bootstrap_func(dataset, irt_model, options, iterations, local_rng):
    """Performs the boostrap sampling."""
    n_responses = dataset.shape[1]

    output_results = list()

    for _ in iterations:
        # Get a sample of RESPONSES
        bootstrap_ndx = local_rng.choice(
            n_responses, size=n_responses, replace=True)
        bootstrap_sample = dataset[:, bootstrap_ndx]

        # solve for parameters
        result = irt_model(bootstrap_sample, options=options)
        output_results.append(result)

    return output_results


def standard_errors_bootstrap(dataset, irt_model, bootstrap_iterations=2000,
                              n_processors=2, solution=None, options=None, seed=None):
    """Computes standard errors of item parameters using bootstrap method.

    This function will be slow, it is best to use multiple processors to decrease
    the processing time.

    Args:
        dataset: Dataset to take random samples from
        irt_model: callable irt function to apply to dataset
        bootstrap_iterations: (int) number of bootstrap resamples to run
        n_processors: (int) number of 
        solution: (optional) parameters from dataset without resampling
        options: dictionary with updates to default options
        seed: (optional) random state integer to reproduce results

Returns:
        solution: parameters from dataset without resampling
        standard_error: parameters from dataset without resampling
        confidence_interval: arrays with 95th percentile confidence intervals
        bias: mean difference of bootstrap mean and solution

    Notes:
        Use partial for irt_models that take a discrimination parameter:
        irt_model = partial(rasch_mml, discrimination=1.2)
        
        !!Graded Unfolding Model and 3PL models are not currently supported!!
    """
    options = validate_estimation_options(options)
    
    seq = np.random.SeedSequence(seed)

    if solution is None:
        solution = irt_model(dataset, options=options)
        
    difficulty_shape = solution['Difficulty'].shape

    # Parallel Random Number Generators
    rngs = [np.random.default_rng(s) for s in seq.spawn(n_processors)]

    bootstrap_chunks = np.array_split(np.arange(bootstrap_iterations, dtype=int), 
                                      n_processors)

    # Run the bootstrap data
    if n_processors > 1:
        with Pool(processes=n_processors) as pool:
            results = pool.starmap(_bootstrap_func, zip(repeat(dataset), repeat(irt_model),
                                                        repeat(options), bootstrap_chunks, rngs))
    else:
        results = starmap(_bootstrap_func, zip(repeat(dataset), repeat(irt_model),
                                               repeat(options), bootstrap_chunks, rngs))
    results = list(results)

    # Unmap the results to compute the metrics
    discrimination_list = []
    difficulty_list = []
    
    # Unmap the results
    for result in results:
        for sub_result in result:
            difficulty_list.append(sub_result['Difficulty'].ravel())
            discrimination_list.append(sub_result['Discrimination'])
    
    # Concatenate the samples
    discrimination_bootstrap = np.vstack(discrimination_list)
    difficulty_bootstrap = np.vstack(difficulty_list)
    
    # Percentiles
    discrimination_ci = [np.percentile(discrimination_bootstrap, 2.5, axis=0), 
                         np.percentile(discrimination_bootstrap, 97.5, axis=0)]
    difficulty_ci = [np.percentile(difficulty_bootstrap, 2.5, axis=0).reshape(difficulty_shape), 
                     np.percentile(difficulty_bootstrap, 97.5, axis=0).reshape(difficulty_shape)]
    
    # Standard Errors
    discrimination_se = np.nanstd(discrimination_bootstrap, axis=0, ddof=1)
    difficulty_se = np.nanstd(difficulty_bootstrap, axis=0, ddof=1).reshape(difficulty_shape)
    
    # Bias
    discrimination_bias = (np.nanmean(discrimination_bootstrap, axis=0) - 
                           solution['Discrimination'])
    
    difficulty_bias = (np.nanmean(difficulty_bootstrap, axis=0) - 
                       solution['Difficulty'].ravel()).reshape(difficulty_shape)
    
    return {
        "Solution": solution,
        "95th CI": {
            "Discrimination": discrimination_ci,
            "Difficulty": difficulty_ci},
        "Standard Errors": {
            'Discrimination': discrimination_se,
            'Difficulty': difficulty_se},
        "Bias":{
            'Discrimination': discrimination_bias,
            'Difficulty': difficulty_bias},            
        }