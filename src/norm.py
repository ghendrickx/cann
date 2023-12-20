_Z_MAX = 5
_Z_MIN = -10


def normalise(z):
    return (z - _Z_MIN) / (_Z_MAX - _Z_MIN)


def reverse(z):
    return _Z_MIN + z * (_Z_MAX - _Z_MIN)
