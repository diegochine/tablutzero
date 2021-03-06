import os
import time


class Map:

    def __init__(self):
        self.atoc = dict()
        self.ctoa = dict()

    def __setitem__(self, a, c):
        self.atoc[a] = c
        self.ctoa[c] = a

    def __getitem__(self, x):
        if x in self.atoc:
            return self.atoc[x]
        else:
            return self.ctoa[x]

    def get_keys(self, dim):
        if dim == 2:
            return self.atoc.keys()
        elif dim == 3:
            return self.ctoa.keys()
        else:
            raise ValueError("dim must be 2 or 3")


class Timeit:
    """
    Decorator class used to log a function's execution time
    """

    def __init__(self, logger=None):
        self.logger = logger

    def __call__(self, f):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            result = f(*args, **kwargs)
            end = time.perf_counter()
            msg = f'{f.__name__} execution took {end - start:.2f} s'
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)
            return result

        return timed


def setup_folders():
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    if 'model' not in os.listdir():
        os.mkdir('model')
        os.mkdir('model/history')
        os.mkdir('model/brain')
