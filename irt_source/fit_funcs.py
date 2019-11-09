from numpy import numpy
from scipy.optimize import optimize

from irt_source import IRTArray
from irt_source import rasch_model, one_parameter_model, two_parameter_model,
                       three_parameter_model


IRTMODELS = {'rasch': rasch_model,
             '1pl': one_parameter_model,
             '2pl': two_parameter_model,
             '3pl': three_parameter_model}


def _min_rasch_model(unknowns, measument):
    pass


class IRTModelFit(object):
    """Class to fit the parameters for an IRT model.

    Args:
        measurement: [Items, Participants] array of binary values
        model: model to use to model the outputs, choose beteween
            'Rasch', '1PL', '2PL', '3PL'

    Attributes:
        parameters: IRTArray of estimated parameters
    """
    def __init__(self, measurement, model="Rasch"):
        """Initialize the IRT estimation class ."""
        self.measurement = measurement

        # Create the output array
        values = [('Ability', measurement.shape[1]),
                  ('Difficulty', measurement.shape[0])]

        if model.lower() == '1pl':
            values.append(('Discrimination', np.ones((1,))))

        if model.lower() in ['2pl', '3pl']:
            values.append(('Discrimination', np.ones((measurement.shape[0],))))

        if model.lower() == '3pl':
            values.append(('Guessing', measurement.shape[0]))

        self.parameters = IRTArray(values)

    def __run__(self):
        """Performs the fit of the data."""
