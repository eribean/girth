#pylint: skip-file
import multiprocessing
import random
from datetime import datetime

from numpy import linspace, ones
from scipy.stats import distributions


_available_models = ["rasch_conditional",
                     "rasch_mml", "rasch_jml", 
                     "onepl_mml", "onepl_jml", 
                     "twopl_mml", "twopl_jml", "grm_jml",
                     "grm_mml", "pcm_mml", "pcm_jml"]


def _set_default(the_dict, the_key, the_default):
    """Updates a dict with key if not found."""
    if the_key not in the_dict.keys():
        the_dict[the_key] = the_default

    if the_dict[the_key] is None:
        the_dict[the_key] = the_default

    return the_dict


def _validate_options(option_dict):
    """Validates options of performance."""

    _set_default(option_dict, "Processor_count", 1)
    _set_default(option_dict, "Chunksize", 20)

    if option_dict["Processor_count"] in [-1, -2]:
        cpu_count = (multiprocessing.cpu_count() // 
                     abs(option_dict["Processor_count"]))
        option_dict["Processor_count"] = cpu_count
    
    return option_dict


def _validate_analysis(analysis_dict):
    """Validates options for analysis."""
    if analysis_dict['Type'] not in ["Dichotomous", "Polytomous"]:
        raise AssertionError(f"{analysis_dict['Type']} not supported")

    # Need to specify model for polytomous
    if ((analysis_dict['Type'] == "Polytomous") and 
        (analysis_dict['SubType'] is None)):
        raise AssertionError("Polytomous Types must have SubType "
                              "defined as ['Graded', 'Credit', 'Unfold']")

    _set_default(analysis_dict, "Seed", random.randint(0, 10000))
    _set_default(analysis_dict,"Trials", 50)

    # Discrimination Defaults
    _set_default(analysis_dict, "Discrimination_pdf", "lognorm")
    _set_default(analysis_dict, "Discrimination_pdf_args", {})
    _set_default(analysis_dict, "Discrimination_count", 10)

    # Difficulty Defaults
    _set_default(analysis_dict, "Difficulty_pdf", "norm")
    _set_default(analysis_dict, "Difficulty_pdf_args", {})
    _set_default(analysis_dict, "Difficulty_count", 10)

    # Ability Defaults
    _set_default(analysis_dict, "Ability_pdf", "norm")
    _set_default(analysis_dict, "Ability_pdf_args", {})
    _set_default(analysis_dict, "Ability_count", [500,])

    if type(analysis_dict["Ability_count"]) is not list:
        raise AssertionError("Ability Counts must be entered as "
                             "a list, i.e.: [100] or [100, 200, ...]")
 
    # If polytomous without fixed parameters, make sure
    # counts is an array of two
    if ((analysis_dict['Type'] == "Polytomous") and 
        (analysis_dict['Difficulty_fixed'] is None) and
        (len(list(analysis_dict['Difficulty_count'])) != 2)):
        raise AssertionError("Difficulty_count with Polytomous Data "
                             "must be a list with two parameters")
    
    return analysis_dict


def _validate_synthesis(synthesis_dict):
    """Validates synthesis parameters."""
    model = synthesis_dict['Model']
    if model not in _available_models:
        raise AssertionError(f"{model} is not a valid model.")
    
    _set_default(synthesis_dict, "Discrimination_pdf", None)
    _set_default(synthesis_dict, "Discrimination_pdf_args", None)
    
    _set_default(synthesis_dict, "Difficulty_pdf", None)
    _set_default(synthesis_dict, "Difficulty_pdf_args", None)

    _set_default(synthesis_dict, "Ability_pdf", "norm")
    _set_default(synthesis_dict, "Ability_pdf_args", 
                    {"loc": 0, "scale": 1})

    return synthesis_dict  


def validate_performance_dict(config_dict):
    """Validates a dictionary of performance parameters.
    
    Args:
        config_dict: dictionary of parameters to run the
                     performance script
    
    Returns:
        dictionary 
    
    Raises:
        Key Error if missing required field
        Assertion Error if something inappropriate is found
    """
    # Give it a name if none given
    now = ("GIRTH_Performance_run" + 
           datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    _set_default(config_dict, "Name", now)

    # Check options
    config_dict['Options'] = _validate_options(config_dict['Options'])
    config_dict['Analysis'] = _validate_analysis(config_dict['Analysis'])

    for key, value in config_dict["Synthesis"].items():
        config_dict["Synthesis"][key] =  _validate_synthesis(value)

    return config_dict


class LinearSpacePDF(object):
    """Fake distribution to return linear space.
    
        Returns a linear space equivalent to 
        np.linspace(loc, loc + scale, size)
    """
    def __init__(self, loc=-3, scale=6):
        self.loc = loc
        self.scale = scale
    
    def rvs(self, size):
        if type(size) is not list:
            size = [size,]

        if len(list(size)) == 1:
            otpt = linspace(self.loc, self.loc+self.scale, size[0])

        else:
            otpt = ones(size)
            x = linspace(self.loc, self.loc+self.scale, size[1])
            otpt *= x[None, :]
        return otpt


def scipy_stats_string_to_functions(pdf_string, pdf_kwargs=None):
    """Return a scipy pdf function from a string

        Visit the link for supported distributions
        https://docs.scipy.org/doc/scipy/reference/stats.html

        Args:
            pdf_string: String to specify a distribution
            pdf_kwargs: Dictionary of distribution parameters
        
        Returns:
            the appropriate scipy function

        Example:
            scipy_stats_string_to_functions("Normal", {"loc": 1.0, "scale": 0.5})
            returns scipy.stats.norm(loc=1.0, scale=0.5)

        Raises:
            Key error if pdf_string not supported
    """
    if pdf_string.lower() == "linear":
        distribution = LinearSpacePDF(**pdf_kwargs)

    else:
        # This will raise a key error if not supported
        distribution = distributions.__dict__[pdf_string.lower()]

        # "Freezes" a distribution for a set of parameters
        if pdf_kwargs is not None:
            distribution = distribution(**pdf_kwargs)
    
    return distribution