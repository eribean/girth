from  copy import deepcopy

import numpy as np


def default_metadata():
    """Creates a default chunk dictionary for distributed arrays.

    Args:
        shape:  full array shape

    Returns:
        chunk dictionary
    """
    return {'names':[], 'shape': [], 'slice': []}


class IRTArray(np.ndarray):
    """Numpy ndarray for running item response theory data.

    This array has metadata associated with to facilitate analysis
    """
    def __new__(cls, input_array, metadata=None):
        if type(input_array) == list: # IRT Array from list of values
            start_index = 0
            data_arrays = []
            metadata = default_metadata()

            for name, other in input_array:
                metadata['names'].append(name)

                if type(other) in [np.ndarray, cls]:
                    data_arrays.append(other)
                    metadata['shape'].append(other.shape[0])
                    metadata['slice'].append(slice(start_index,
                                                   start_index + other.shape[0]))
                    start_index += other.shape[0]

                else: # Integers
                    data_arrays.append(np.zeros((other,)))
                    metadata['shape'].append(other)
                    metadata['slice'].append(slice(start_index,
                                                   start_index + other))
                    start_index += other

            obj = np.concatenate(data_arrays).view(cls)

        else: # IRT Array from basic array
            obj = input_array.view(cls)

        if metadata is None:
            metadata = default_metadata()
        obj.metadata = deepcopy(metadata)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, 'metadata', default_metadata())

    def __reduce__(self):
        """Used for pickling purpose."""
        pickled_state = super(IRTArray, self).__reduce__()
        new_state = pickled_state[2] + (self.metadata,)
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        """Restores from pickle."""
        self.metadata = state[-1]
        super(IRTArray, self).__setstate__(state[0:-1])

    def __getitem__(self, key):
        if key in self.names:
            return_obj = self[self.slices[key]]
        else:
            return_obj = super(IRTArray, self).__getitem__(key)

        return return_obj

    def __setitem__(self, key, value):
        if key in self.names:
            self[self.slices[key]] = value
        else:
            super(IRTArray, self).__setitem__(key, value)

    def set_metadata_dict(self, new_dict):
        self.metadata = deepcopy(new_dict)

    def get_metadata_dict(self):
        return deepcopy(self.metadata)

    @property
    def names(self):
        return self.metadata['names']

    @property
    def shapes(self):
        return {key: value for (key, value) in zip(self.names, self.metadata['shape'])}

    @property
    def slices(self):
        return {key: value for (key, value) in zip(self.names, self.metadata['slice'])}
