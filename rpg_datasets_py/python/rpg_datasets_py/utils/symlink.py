import os
from os.path import dirname


def symlink(name):
    return os.path.join(dirname(dirname(dirname(dirname(__file__)))), name)
