import numpy as np

import config as cfg
import loggers as lg
from MCTS import MCTS, Node
from game import MAP
from neuralnet import ResidualNN


class Player:

    def __init__(self, color, name, nnet,
                 timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS, c_puct=cfg.CPUCT):
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
        self.c_puct = c_puct
        self.game_over = False
        self.brain: ResidualNN = nnet

    def build_mcts(self, state, p_root):
        self.mcts = MCTS(self.color, Node(state), p_root, self.c_puct)

    def act(self, state):
        """ computes best action based on given state
        """
        v, p = self.brain.predict(state)
        if self.mcts is None or hash(state) not in self.mcts.tree:
            self.build_mcts(state, p)
        else:
            self.mcts.change_root(state, p)

        # time to roll
        for sim in range(self.simulations):
            self.simulate()

        action = self.mcts.choose_action()

        return action, None

    def simulate(self) -> None:
        """
        Performs one monte carlo simulation, using the neural network to evaluate the leaves
        """
        lg.logger_player.info('PERFORMING SIMULATION')
        # selection
        leaf, path = self.mcts.select_leaf()
        v, p = self.brain.predict(leaf.state)
        # expansion
        self.mcts.expand_leaf(leaf, p)
        # backpropagation
        self.mcts.backpropagation(v, path)

    def replay(self, memories) -> None:
        """
        Retrain the network using the given memories
        :param memories: iterable of memories, i.e. objects with attributes 'state', ' value', 'pi
        """
        lg.logger_player.info('RETRAINING MODEL')

        for i in range(cfg.TRAINING_LOOPS):
            minibatch = np.random.sample(memories, min(cfg.BATCH_SIZE, len(memories)))

            X = np.array([memory['state'].convert_into_cnn() for memory in minibatch])
            y = {'value_head': np.array([memory['value'] for memory in minibatch]),
                 'policy_head': np.array([memory['pi'] for memory in minibatch])}

            loss = self.brain.fit(X, y, epochs=cfg.EPOCHS, verbose=cfg.VERBOSE, validation_split=0, batch_size=32)
            lg.logger_player.info('ITERATION {:3d}/{:3d}, LOSS {}'.format(i, cfg.TRAINING_LOOPS, loss.history))
