import numpy as np
from itertools import starmap, repeat
from functools import partial


__all__ = ["standard_errors_bootstrap"]


def _bootstrap_func(dataset, irt_model, options, iterations, local_seed):
    """Performs the boostrap sampling."""
    np.random.seed(local_seed)

    n_responses = dataset.shape[1]

    output_results = list()

    for _ in range(iterations[0], iterations[1]):
        # Get a sample of RESPONSES
        bootstrap_ndx = np.random.choice(
            n_responses, size=n_responses, replace=True)
        bootstrap_sample = dataset[:, bootstrap_ndx]

        # solve for parameters
        result = irt_model(bootstrap_sample, options=options)
        output_results.append(result)

    return output_results


def standard_errors_bootstrap(dataset, irt_model, solution=None, options=None, seed=None):
    """Computes standard errors of item parameters using bootstrap method.
    This function will be sloooow, it is best to use multiple processors to decrease
    the processing time.
    Args:
        dataset: Dataset to take random samples from
        irt_model: callable irt function to apply to dataset
        solution: (optional) parameters from dataset without resampling
        options: dictionary with updates to default options
        seed: (optional) random state integer to reproduce results
    Returns:
        solution: parameters from dataset without resampling
        standard_error: parameters from dataset without resampling
        confidence_interval: arrays with 95th percentile confidence intervals
        bias: mean difference of bootstrap mean and solution
    Options:
        * n_processors: int
        * bootstrap_iterations: int
    Notes:
        Use partial for irt_models that take an discrimination parameter:
        irt_model = partial(rasch_mml, discrimination=1.2)
    """
    options = validate_estimation_options(options)

    if seed is None:
        seed = np.random.randint(0, 100000, 1)[0]

    if solution is None:
        solution = irt_model(dataset, options=options)

    n_processors = options['n_processors']
    bootstrap_iterations = options['bootstrap_iterations']
    chunksize = np.linspace(0, bootstrap_iterations,
                            n_processors + 1, dtype='int')
    chunksize = list(zip(chunksize[:-1], chunksize[1:]))
    seeds = seed * np.arange(1.0, len(chunksize)+1, dtype='int')

    map_func = starmap
    if n_processors > 1:
        map_func = Pool(processes=n_processors).starmap

    # Run the bootstrap data
    results = map_func(_bootstrap_func, zip(repeat(dataset), repeat(irt_model),
                                            repeat(options), chunksize, seeds))
    results = list(results)

    # Unmap the results to compute the metrics
    ses_list = list()
    ci_list = list()
    bias_list = list()

    for p_ndx, parameter in enumerate(solution):
        temp_result = np.concatenate([list(zip(*results[ndx]))[p_ndx]
                                      for ndx in range(len(results))])

        parameter = np.atleast_1d(parameter)

        bias_list.append(np.nanmean(temp_result, axis=0) - parameter)
        ses_list.append(np.nanstd(temp_result, axis=0, ddof=0))
        
        if parameter.shape[0] == 1:
            ci_list.append((np.percentile(temp_result, 2.5, axis=0),
                            np.percentile(temp_result, 97.5, axis=0)))
        else:
            ci_list.append(list(zip(np.percentile(temp_result, 2.5, axis=0),
                                    np.percentile(temp_result, 97.5, axis=0))))

    return {'Solution': solution,
            'Standard Error': ses_list,
            'Confidence Interval': ci_list,
            'Bias': bias_list}