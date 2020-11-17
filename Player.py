import numpy as np

from src.pytablut.game import State
from src.pytablut.MCTS import MCTS


class Player:

    def __init__(self, color, name, timeout=60, algo='mcts'):
        """
        :param color: color of the player, either BLACK or WHITE
        :param name: name of the player
        :param timeout: timeout in seconds for each move computation
        """
        self.name = name
        self.color = color.upper()
        if self.color not in ('BLACK', 'WHITE'):
            raise ValueError('wrong color, must either BLACK or WHITE ')
        self.timeout = timeout
        if self.timeout <= 0:
            raise ValueError('timeout must be >0')
        if algo == 'mcts':
            self.algo = None
        else:
            raise ValueError('wrong algo parameter')
        self.board = None
        self.game_over = False
        self.turn = None

    def compute_move(self, state):
        """ computes moves based on given state"""
        best_move = self.algo.compute_move(state)
        return best_move
