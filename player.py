from multiprocessing import Process
import numpy as np

import config as cfg
import loggers as lg
from MCTS import MCTS, Node
from game import MAP
from neuralnet import ResidualNN


class Player:

    def __init__(self, color, name, nnet, turns_before_tau0=cfg.TURNS_BEFORE_TAU0, tau=cfg.TAU,
                 timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS, c_puct=cfg.CPUCT):
        """
        :param color: color of the player, either BLACK or WHITE
        :param name: name of the player
        :param timeout: timeout in seconds for each move computation
        """
        self.name = name
        self.color: int = MAP[color]
        self.timeout: int = timeout
        self.mcts: MCTS = None
        self.brain: ResidualNN = nnet
        self.simulations: int = simulations
        self.c_puct: int = c_puct
        self.game_over: bool = False
        self.turns_before_tau0 = turns_before_tau0
        self.tau = tau
        self.turn = 1

    def reset(self):
        self.turn = 1

    def build_mcts(self, state, p):
        """"""
        lg.logger_player.info("BUILDING MCTS, P VALUES:")
        lg.logger_player.info(p)
        if self.mcts is None or hash(state) not in self.mcts.tree:
            self.mcts = MCTS(self.color, Node(state), p, self.c_puct)
        else:
            self.mcts.new_root(state, p)

    def act(self, state):
        """
        computes best action based on given state
        :return tuple (action, pi)
        """
        lg.logger_player.info("COMPUTING ACTION FOR STATE {}".format(state.id))
        v, p = self.brain.predict(state)
        self.build_mcts(state, p)

        lg.logger_player.info("size of the tree (start): {}".format(len(self.mcts.tree)))
        self.simulate()
        lg.logger_player.info("size of the tree (end)  : {}".format(len(self.mcts.tree)))

        action, pi = self.choose_action()
        self.turn += 1
        return action, pi

    def choose_action(self) -> tuple:
        """
        Chooses the best action from the current state,
        either deterministically or stochastically
        :return: tuple (action, pi), pi are normalized probabilities
        """
        pi = np.array([edge.N for edge in self.mcts.root.edges])
        if self.turn >= self.turns_before_tau0:
            act_idx = np.argmax(pi)
            action = self.mcts.root.edges[act_idx].action
        else:
            pvals = np.power(pi, 1 / self.tau)
            pvals = pvals / np.sum(pvals)  # normalization, not sure it's actually needed
            act_idx = np.argwhere(np.random.multinomial(1, pvals) == 1).reshape(-1)
            action = self.mcts.root.edges[act_idx[0]].action
        lg.logger_player.info('COMPUTED ACTION: {}'.format(action))
        return action, (pi/pi.sum())

    @profile
    def simulate(self) -> None:
        """
        Performs the monte carlo simulations, using the neural network to evaluate the leaves
        """
        for sim in range(self.simulations):
            if sim % 50 == 0:
                lg.logger_player.info('{:3d} SIMULATIONS PERFORMED'.format(sim))
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
            minibatch = np.random.choice(memories, min(cfg.BATCH_SIZE, len(memories)))
            pi = [self.brain.map_into_action_space(actions, pi)
                  for actions, pi in zip([memory['state'].actions for memory in minibatch],
                                         [memory['pi'] for memory in minibatch])]
            X = np.array([memory['state'].convert_into_cnn() for memory in minibatch])
            y = {'value_head': np.array([memory['value'] for memory in minibatch]),
                 'policy_head': np.array(pi)}

            loss = self.brain.fit(X, y, epochs=cfg.EPOCHS, verbose=cfg.VERBOSE,
                                  validation_split=0, batch_size=minibatch.size)
            lg.logger_player.info('ITERATION {:3d}/{:3d}, LOSS {}'.format(i, cfg.TRAINING_LOOPS, loss.history))
