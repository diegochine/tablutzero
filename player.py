import numpy as np

import pytablut.config as cfg
import pytablut.loggers as lg
from pytablut.MCTS import MCTS, Node
from pytablut.game import MAP
from pytablut.neuralnet import ResidualNN


class Player:

    def __init__(self, color, name, nnet, timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS):
        """
        :param color: color of the player, either BLACK or WHITE
        :param name: name of the player
        :param timeout: timeout in seconds for each move computation
        """
        self.name = name
        if color not in ('BLACK', 'WHITE'):
            raise ValueError('wrong color, must either BLACK or WHITE ')
        self.color = MAP[color]
        self.timeout = timeout
        if self.timeout <= 0:
            raise ValueError('timeout must be >0')
        self.mcts = None
        self.simulations = simulations
        self.game_over = False
        self.brain : ResidualNN = nnet

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

    def simulate(self) -> None:
        # selection
        leaf, path = self.mcts.select_leaf()
        v, p, action_map = self.brain.predict(leaf.state)
        # expansion
        self.mcts.expand_leaf(leaf, p, action_map)
        # backpropagation
        self.mcts.backpropagation(v, path)

    def replay(self, memories):
        lg.logger_player.info('******RETRAINING MODEL******')

        for i in range(cfg.TRAINING_LOOPS):
            minibatch = np.random.sample(memories, min(cfg.BATCH_SIZE, len(memories)))

            X = np.array([memory['state'].convert_into_cnn() for memory in minibatch])
            y = {'value_head': np.array([memory['value'] for memory in minibatch]),
                 'policy_head': np.array([memory['pi'] for memory in minibatch])}

            loss = self.brain.fit(X, y, epochs=cfg.EPOCHS, verbose=cfg.VERBOSE, validation_split=0)
            lg.logger_player.info('NEW LOSS {}'.format(loss.history))
