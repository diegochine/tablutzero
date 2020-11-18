import numpy as np

from src.pytablut.MCTS import MCTS, Node


class Player:

    def __init__(self, color, name, timeout=60, simulations=160):
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
        self.mcts = None
        self.simulations = simulations
        self.game_over = False

    def build_mcts(self, state):
        self.mcts = MCTS(self.color, Node(state))

    def act(self, state):
        """ computes best action based on given state"""
        if self.mcts is None or hash(state) not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.mcts.change_root(state)

        # time to roll
        for sim in range(self.simulations):
            self.simulate()

        action = self.mcts.choose_action()

        return action

    def simulate(self):
        # selection
        leaf, path = self.mcts.select_leaf()
        # expansion
        self.mcts.expand_leaf(leaf)
        # random playout
        score = self.mcts.random_playout(leaf)
        # backpropagation
        self.mcts.backpropagation(score, path)
