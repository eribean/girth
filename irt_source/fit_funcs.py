from sys import float_info
from copy import deepcopy

import numpy as np
from scipy.optimize import basinhopping

from irt_source import IRTArray
from irt_source import (rasch_model, one_parameter_model, two_parameter_model,
                        three_parameter_model)


IRTMODELS = {'rasch': rasch_model,
             '1pl': one_parameter_model,
             '2pl': two_parameter_model,
             '3pl': three_parameter_model}


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
        self.measurement = measurement.astype('bool')
        self.model = model.lower()
        self.probability_func = IRTMODELS[self.model]

        # Create the output array
        values = [('Abilities', measurement.shape[1]),
                  ('Difficulty', measurement.shape[0])]

        if self.model == '1pl':
            values.append(('Discrimination', np.ones((1,))))

        if self.model in ['2pl', '3pl']:
            values.append(('Discrimination', np.ones((measurement.shape[0],))))

        if self.model == '3pl':
            values.append(('Guessing', measurement.shape[0]))

        self.parameters = IRTArray(values)

    def _expectation_step(self):
        """Maximizes abilities over given parameters"""
        local_values = deepcopy(self.parameters)

        def local_min(abilities):
            local_values['Abilities'] = abilities
            probabilities = self.probability_func(local_values)
            probabilities = np.where(self.measurement, probabilities,
                                     1.0 - probabilities)

            return -1 * np.log(probabilities).sum()

        minimizer_kwargs = {'method': 'SLSQP'}
        result = basinhopping(local_min, self.parameters['Abilities'],
                              niter_success=10, minimizer_kwargs=minimizer_kwargs)

        self.parameters['Abilities'] = (result.x - result.x.mean()) / result.x.std(ddof=1)

    def _maximization_step(self):
        """Maximizes parameters over given abilities"""
        local_values = deepcopy(self.parameters)

        # Create an array for parameters
        model_parameters = [*self.parameters.shapes]
        model_parameters.pop(model_parameters.index('Abilities'))
        model_array = IRTArray([(key, self.parameters.shapes[key])
                                for key in model_parameters])
        model_slices = model_array.slices

        for key in model_parameters:
            model_array[key] = self.parameters[key]

        def local_min(estimation_parameters):
            for key in model_parameters:
                local_values[key] = estimation_parameters[model_slices[key]]

            probabilities = self.probability_func(local_values)
            probabilities = np.where(self.measurement, probabilities,
                                     1.0 - probabilities)

            return -1 * np.log(probabilities + float_info.epsilon).sum()

        minimizer_kwargs = {'method': 'SLSQP'}
        result = basinhopping(local_min, model_array, niter_success=10,
                              minimizer_kwargs=minimizer_kwargs)

        for key in model_parameters:
            self.parameters[key] = result.x[model_slices[key]]

        # Apply Constraints
        self.parameters['Difficulty'] -= self.parameters['Difficulty'].mean()
        #
        # if self.model in ['2pl', '3pl']:
        #     self.parameters['Discrimination'] /= self.parameters['Discrimination'].prod()

    def __run__(self):
        """Performs the fit of the data."""
