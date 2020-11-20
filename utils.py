class Map:

    def __init__(self):
        self.atob = dict()
        self.btoa = dict()

    def __setitem__(self, a, b):
        self.atob[a] = b
        self.btoa[b] = a

    def __getitem__(self, x):
        if x in self.atob:
            return self.atob[x]
        else:
            return self.btoa[x]
