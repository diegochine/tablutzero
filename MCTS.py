import numpy as np

import config as cfg
import loggers as lg
from game import State


class Node:

    def __init__(self, state):
        """
        each node of represents a state
        :param state: state
        """
        self.state: State = state
        self.id: int = hash(state)
        self.edges: list = []

    def is_leaf(self) -> bool:
        return len(self.edges) == 0


class Edge:

    def __init__(self, in_node: Node, out_node: Node, action, p):
        """
        each edge represents an action from a state to another
        :param in_node: node of the initial state
        :param out_node: node of the next state
        :param action: the action
        """
        self.in_node: Node = in_node
        self.out_node: Node = out_node
        self.action: tuple = action
        self.N = 0   # number of times action has been taken from initial state
        self.W = 0.  # total value of next state
        self.Q = 0.  # mean value of next state
        self.P = p   # prior probability of selecting this action


class MCTS:

    def __init__(self, player, root: Node, p_root, c_puct: float = cfg.CPUCT):
        self.player = player
        self.root: Node = root
        self.tree = {root.id: root}
        self.c_puct = c_puct
        self.new_root(self.root.state, p_root)

    def new_root(self, state: State, p) -> None:
        if self.root.state.id != state.id:
            tmp = self.root
            self.root = self.tree[state.id]
            for edge in tmp.edges:
                self._delete_subtree(edge)
        if self.root.is_leaf():
            self.expand_leaf(self.root, p)

    def add_node(self, node: Node):
        self.tree[node.id] = node

    def select_leaf(self) -> (Node, list):
        lg.logger_mcts.info('SELECTING LEAF')
        node = self.root
        path = []

        while not node.is_leaf():
            max_QU = -np.inf
            N = np.sum([edge.N for edge in node.edges])
            simulation_edge = None
            lg.logger_mcts.debug('PLAYER TURN {}'.format(node.state.turn))

            if node.id == self.root.id:
                epsilon = cfg.EPSILON
                nu = np.random.dirichlet([cfg.ALPHA] * len(node.edges))
            else:
                epsilon = 0
                nu = [0] * len(node.edges)

            for i, edge in enumerate(node.edges):
                #lg.logger_mcts.debug('EVALUATING ACTION: {}'.format(edge.action))

                U = self.c_puct * \
                    ((1 - epsilon) * edge.P + epsilon * nu[i]) * \
                    np.sqrt(N) / (1 + edge.N)
                Q = edge.Q
                #lg.logger_mcts.debug('Q: {}, U: {}'.format(Q, U))

                if Q+U > max_QU and edge not in path:
                    lg.logger_mcts.debug('UPDATING SIMULATION EDGE')
                    max_QU = Q+U
                    simulation_edge = edge

            # next_state = node.state.transition_function(simulation_edge.action)
            node = simulation_edge.out_node
            path.append(simulation_edge)

        return node, path

    def expand_leaf(self, leaf: Node, p):
        lg.logger_mcts.info('EXPANDING LEAF WITH ID {}'.format(leaf.id))
        for action in leaf.state.get_actions():
            next_state = leaf.state.transition_function(action)
            if next_state.id not in self.tree:
                new_leaf = Node(next_state)
                self.add_node(new_leaf)
                new_edge = Edge(leaf, new_leaf, action, p[action])
                leaf.edges.append(new_edge)

    def random_playout(self, leaf: Node):
        lg.logger_mcts.info('PERFORMING RANDOM PLAYOUT')
        state = leaf.state
        while not state.is_terminal:
            acts = state.get_actions()
            # FIXME sometimes during random playout we have 0 actions and it crashes (low >= high)
            rnd_a = acts[np.random.randint(0, len(acts))]
            state = state.transition_function(rnd_a)
        return state.value

    def backpropagation(self, v: float, path: list):
        lg.logger_mcts.info('PERFORMING BACKPROPAGATION WITH v = {:.2f}'.format(v[0]))
        direction = -1
        for edge in path:
            edge.N += 1
            edge.W += v * direction
            direction *= -1
            edge.Q = edge.W / edge.N
            lg.logger_mcts.info('Act = {}, N = {}, W = {}, Q = {}'.format(edge.action, edge.N, edge.W, edge.Q))

    def _delete_subtree(self, edge):
        node = edge.out_node
        if node != self.root:
            for edge in node.edges:
                self._delete_subtree(edge)
        try:
            del self.tree[node.id]
        except KeyError:
            pass
