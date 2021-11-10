from scipy.stats import norm as gaussian


__all__= ['default_options', 'validate_estimation_options']


def default_options():
    """ Dictionary of options used in Girth.

    Args:
        max_iteration: [int] maximum number of iterations
            allowed during processing. (Default = 25)
        distribution: [callable] function that returns a pdf
            evaluated at quadrature points, p = f(theta).
            (Default = scipy.stats.norm(0, 1).pdf)
        quadrature_bounds: (lower, upper) bounds to limit
            numerical integration. Default = (-5, 5)
        quadrature_n: [int] number of quadrature points to use
                        Default = 61
        hyper_quadrature_n: [int] number of quadrature points to use to
                            estimate hyper prior integral (only mml_eap)
        use_LUT: [boolean] use a look up table in mml functions
        estimate_distribution: [boolean] estimate the latent distribution
                               using cubic splines
        number_of_samples: [int] number of samples to use when
                           estimating distribuion, must be > 5
        initial_guess: [boolean] seed the convergence with an initial guess
                       Default = True
        num_processors: [int] number of cores to use. Default=1
    """
    return {"max_iteration": 25,
            "distribution": gaussian(0, 1).pdf,
            "quadrature_bounds": (-4.5, 4.5),
            "quadrature_n": 41,
            "hyper_quadrature_n": 41,
            "use_LUT": True,
            "estimate_distribution": False,
            "number_of_samples": 9,
            "initial_guess": True,
            "num_processors": 1
            }


def validate_estimation_options(options_dict=None):
    """ Validates an options dictionary.

    Args:
        options_dict: Dictionary with updates to default_values

    Returns:
        options_dict: Updated dictionary

    """
    validate = {'max_iteration':
                    lambda x: isinstance(x, int) and x > 0,
                'distribution':
                    callable,
                'quadrature_bounds':
                    lambda x: isinstance(x, (tuple, list)) and (x[1] > x[0]),
                'quadrature_n':
                    lambda x: isinstance(x, int) and x > 7,
                'hyper_quadrature_n':
                    lambda x: isinstance(x, int) and x > 7,                    
                'use_LUT':
                    lambda x: isinstance(x, bool),
                'estimate_distribution':
                    lambda x: isinstance(x, bool),
                "number_of_samples": 
                    lambda x: isinstance(x, int) and x >= 5,
                "initial_guess":
                    lambda x: isinstance(x, bool),
                "num_processors":
                    lambda x: isinstance(x, int)
                }
    
    # A complete options dictionary
    full_options = default_options()
    
    if options_dict:
        if not isinstance(options_dict, dict):
            raise AssertionError("Options must be a dictionary got: "
                                f"{type(options_dict)}.")

        for key, value in options_dict.items():
            if not validate[key](value):
                raise AssertionError("Unexpected key-value pair: "
                                     f"{key}: {value}. Please see "
                                     "documentation for expected inputs.")

        full_options.update(options_dict)

    return full_options