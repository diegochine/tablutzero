import numpy as np

import config as cfg
import loggers as lg
from MCTS import MCTS, Node
from game import MAP
from neuralnet import ResidualNN


class Player:

    def __init__(self, color, name, timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS, c_puct=cfg.CPUCT):
        """
        :param color: color of the player, either BLACK or WHITE
        :param name: name of the player
        :param timeout: timeout in seconds for each move computation
        """
        self.name = name
        self.color: int = MAP[color]
        self.timeout: int = timeout
        self.mcts: MCTS = None
        self.simulations: int = simulations
        self.c_puct: int = c_puct
        self.game_over: bool = False
        self.turn = 1

    def reset(self):
        self.turn = 1
        if self.mcts is not None:
            self.mcts.delete_tree()
            self.mcts = None

    def build_mcts(self, state):
        """"""
        lg.logger_player.info("BUILDING MCTS")
        if self.mcts is None or hash(state) not in self.mcts.tree:
            self.mcts = MCTS(self.color, Node(state), self.c_puct)
        else:
            self.mcts.new_root(state)

    def act(self, state):
        """
        computes best action based on given state
        :return tuple (action, pi)
        """
        lg.logger_player.info("COMPUTING ACTION FOR STATE {}".format(state.id))
        self.build_mcts(state)

        lg.logger_player.info("size of the tree (start): {}".format(len(self.mcts.tree)))
        self.simulate()
        lg.logger_player.info("size of the tree (end)  : {}".format(len(self.mcts.tree)))

        action, pi = self.choose_action()
        self.turn += 1
        return action, pi

    def choose_action(self) -> tuple:
        """
        Chooses the best action from the current state using ucb1
        :return: tuple (action, pi), pi are normalized probabilities
        """

        pi = np.array([edge.N for edge in self.mcts.root.edges])
        choices_weights = [
            (e.W / e.N) + self.c_puct * np.sqrt((2 * np.log(pi.sum()) / e.N))
            for e in self.mcts.root.edges
        ]

        act_idx = np.argmax(choices_weights)
        action = self.mcts.root.edges[act_idx].action

        lg.logger_player.info('COMPUTED ACTION: {}'.format(action))
        return action, (pi / pi.sum())


    def simulate(self) -> None:
        """
        Performs the monte carlo simulations
        """
        for sim in range(self.simulations):
            if sim % 50 == 0:
                lg.logger_player.info('{:3d} SIMULATIONS PERFORMED'.format(sim))
            # selection
            leaf, path = self.mcts.select_leaf()
            # expansion
            self.mcts.expand_leaf(leaf)
            # random playouts
            v = self.mcts.random_playout(leaf)
            # backpropagation
            self.mcts.backpropagation(v, path)
