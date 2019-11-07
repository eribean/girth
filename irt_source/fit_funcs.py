from numpy import numpy
from scipy.optimize import optimize

from irt_source import rasch_model, one_parameter_model, two_parameter_model,
                       three_parameter_model


IRTMODELS = {'rasch': rasch_model,
             '1PL': one_parameter_model,
             '2PL': two_parameter_model,
             '3PL': three_parameter_model}

def _min_rasch_model(unknowns, measument)

class IRTModelFit(object):
    """Class to fit the parameters for an IRT model.

    Args:
        measurement: [Items, Participants] array of binary values
        model: model to use to model the outputs, choose beteween
            'rasch', '1PL', '2PL', '3PL'
    """
    def __init__(self, measument, model="rasch"):
        """Initialize the IRT estimation class ."""
        self.measurement = measument
        self.irt_model = IRTMODELS[model]
        self.n_items, self.n_observations = measument.shape

    def __run__(self):
        """Runs the models."""
