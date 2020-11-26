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
        self.turns_before_tau0 = turns_before_tau0
        self.tau = tau
        self.turn = 1

    def reset(self):
        self.turn = 1
        if self.mcts is not None:
            self.mcts.delete_tree()
            self.mcts = None

    def build_mcts(self, state):
        """"""
        lg.logger_player.info("BUILDING MCTS")
        if self.mcts is None:
            self.mcts = MCTS(self.color, Node(state), self.c_puct)
        else:
            self.mcts.new_root(Node(state))

    def act(self, state):
        """
        computes best action based on given state
        :return tuple (action, pi)
        """
        lg.logger_player.info("COMPUTING ACTION FOR STATE {}".format(state.id))
        # v = self.brain.predict(state)
        self.build_mcts(state)
        self.simulate()
        action = self.choose_action()
        self.turn += 1
        return action

    def choose_action(self) -> tuple:
        """
        Chooses the best action from the current state,
        either deterministically or stochastically
        :return: tuple (action, pi), pi are normalized probabilities
        """
        pi = np.array([edge.Q for edge in self.mcts.root.edges])
        if self.turn >= self.turns_before_tau0 or True:
            act_idx = np.argmax(pi)
            action = self.mcts.root.edges[act_idx].action
        else:  # FIXME testare nuove cose
            pvals = np.power(pi, 1 / self.tau)
            pvals = pvals / np.sum(pvals)  # normalization, not sure it's actually needed
            act_idx = np.argwhere(np.random.multinomial(1, pvals) == 1).reshape(-1)[0]
            action = self.mcts.root.edges[act_idx].action
        lg.logger_player.info('COMPUTED ACTION: {}'.format(action))
        self.mcts.new_root(self.mcts.root.edges[act_idx].out_node)
        return action

    def simulate(self) -> None:
        """
        Performs the monte carlo simulations, using the neural network to evaluate the leaves
        """
        for sim in range(self.simulations):
            if sim % 50 == 0:
                lg.logger_player.info('{:3d} SIMULATIONS PERFORMED'.format(sim))
            # selection
            leaf, path = self.mcts.select_leaf()
            v = self.brain.predict(leaf.state)
            # expansion
            found_terminal = self.mcts.expand_leaf(leaf)
            if found_terminal:
                if leaf.state.turn == self.color:
                    v = 1
                else:
                    v = -1
            # backpropagation
            self.mcts.backpropagation(v, path)

    def replay(self, memories) -> None:
        """
        Retrain the network using the given memories
        :param memories: iterable of memories, i.e. objects with attributes 'state', ' value', 'turn'
        """
        lg.logger_player.info('RETRAINING MODEL')

        for i in range(cfg.TRAINING_LOOPS):
            minibatch = np.random.choice(memories, min(cfg.BATCH_SIZE, len(memories)))
            X = np.array([memory['state'].convert_into_cnn() for memory in minibatch])
            y = np.array([memory['value'] for memory in minibatch])

            loss = self.brain.fit(X, y, epochs=cfg.EPOCHS, verbose=cfg.VERBOSE,
                                  validation_split=0, batch_size=minibatch.size)
            lg.logger_nnet.info('ITERATION {:3d}/{:3d}, LOSS {}'.format(i, cfg.TRAINING_LOOPS, loss.history))
