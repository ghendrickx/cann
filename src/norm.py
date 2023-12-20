"""
Normalisation functions (min-max normalisation).

Author: Gijs G. Hendrickx
"""
import numpy as np

_Z_MAX = 5
_Z_MIN = -10


def normalise(z: np.ndarray) -> np.ndarray:
    """Normalise z-values according the min-max scaling.

    :param z: non-normalised z-values
    :type z: numpy.ndarray

    :return: normalised z-values
    :rtype: numpy.ndarray
    """
    return (z - _Z_MIN) / (_Z_MAX - _Z_MIN)


def reverse(z: np.ndarray) -> np.ndarray:
    """Reverse-normalise z-values according the min-max scaling.

    :param z: normalised z-values
    :type z: numpy.ndarray

    :return: non-normalised z-values
    :rtype: numpy.ndarray
    """
    return _Z_MIN + z * (_Z_MAX - _Z_MIN)
