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
            raise ValueError("dim must be either 2 o 3")

