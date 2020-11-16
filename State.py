import numpy as np


class State:

    def __init__(self, board, turn):
        self.board = board
        self.turn = turn

    def __hash__(self):
        return hash((tuple(tuple(row) for row in self.board), self.turn))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_checkers(self):
        """ return positions of checkers that can be moved in this state """
        checkers = set(tuple(x) for x in np.argwhere(self.board == self.turn))
        if self.turn == 'WHITE':
            checkers.add(tuple(np.argwhere(self.board == 'KING')[0]))
        return checkers
