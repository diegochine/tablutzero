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

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def is_leaf(self) -> bool:
        return len(self.edges) == 0


class Edge:

    def __init__(self, in_node: Node, out_node: Node, action):
        """
        each edge represents an action from a state to another
        :param in_node: node of the initial state
        :param out_node: node of the next state
        :param action: the action
        """
        self.in_node: Node = in_node
        self.out_node: Node = out_node
        self.action: tuple = action
        self.N = 0  # number of times action has been taken from initial state
        self.W = 0.  # total value of next state
        self.Q = 0.  # mean value of next state


class MCTS:

    def __init__(self, player, root: Node, c_puct: float = cfg.CPUCT):
        self.player = player
        self.root: Node = root
        self.c_puct = c_puct
        self.new_root(self.root)

    def _delete_subtree(self, edge):
        node = edge.out_node
        del edge.out_node
        del edge.in_node
        del edge
        for out_edge in node.edges:
            self._delete_subtree(out_edge)
        del node.edges
        del node

    def delete_tree(self):
        for edge in self.root.edges:
            self._delete_subtree(edge)
        del self.root.edges
        self.root = None

    def new_root(self, node: Node):
        if self.root != node:
            tmp = self.root
            self.root = node
            for edge in [edge for edge in tmp.edges if edge.out_node != self.root]:
                self._delete_subtree(edge)
            del tmp.edges
        if self.root.is_leaf():
            self.expand_leaf(self.root)
        any_terminal = np.argwhere([edge.out_node.state.is_terminal for edge in self.root.edges])
        if np.any(any_terminal):
            return self.root.edges[any_terminal[0, 0]].action
        else:
            return None

    def select_leaf(self) -> (Node, list):
        lg.logger_mcts.info('SELECTING LEAF')
        node = self.root
        path = []

        while not node.is_leaf():
            max_QU = -np.inf
            Np = np.sum([edge.N for edge in node.edges])
            simulation_edge = None
            lg.logger_mcts.debug('PLAYER TURN {}'.format(node.state.turn))

            if node == self.root:
                nu = np.random.dirichlet([cfg.ALPHA] * len(node.edges))
            else:
                nu = [0] * len(node.edges)

            for i, edge in enumerate(node.edges):
                if edge.N == 0:
                    U = np.inf
                else:
                    U = self.c_puct * np.sqrt(np.log(Np) / edge.N)

                # U = self.c_puct * ((1 - epsilon) * edge.P + epsilon * nu[i]) * np.sqrt(N) / (1 + edge.N)
                QU = edge.Q + U
                lg.logger_mcts.info('ACTION: {}, QU: {}'.format(edge.action, QU))

                if QU > max_QU and edge not in path:
                    lg.logger_mcts.debug('UPDATING SIMULATION EDGE')
                    max_QU = QU
                    simulation_edge = edge

            # next_state = node.state.transition_function(simulation_edge.action)
            node = simulation_edge.out_node
            path.append(simulation_edge)

        return node, path

    def expand_leaf(self, leaf: Node) -> bool:
        lg.logger_mcts.info('EXPANDING LEAF WITH ID {}'.format(leaf.id))
        found_terminal = leaf.state.is_terminal
        for action in leaf.state.actions:
            next_state = leaf.state.transition_function(action)
            new_leaf = Node(next_state)
            new_edge = Edge(leaf, new_leaf, action)
            leaf.edges.append(new_edge)
            if next_state.is_terminal:
                found_terminal = True
        return found_terminal

    def random_playout(self, leaf: Node, check_terminals):
        lg.logger_mcts.info('PERFORMING RANDOM PLAYOUT')
        current_state = leaf.state
        path = []
        while not current_state.is_terminal:
            # FIXME sometimes during random playout we have 0 actions and it crashes (low >= high)
            if check_terminals:
                next_states = ([current_state.transition_function(act) for act in current_state.actions])
                any_terminal = np.argwhere([state.is_terminal for state in next_states])
                if np.any(any_terminal):
                    act_idx = any_terminal[0, 0]
                else:
                    act_idx = np.random.randint(0, len(current_state.actions))
                path.append(current_state.actions[act_idx])
                current_state = next_states[act_idx]
            else:
                act_idx = np.random.randint(0, len(current_state.actions))
                path.append(current_state.actions[act_idx])
                current_state = current_state.transition_function(current_state.actions[act_idx])

        if current_state.turn != self.player:
            return 1, path
        else:
            return -1, path

    def backpropagation(self, v, path: list):
        lg.logger_mcts.info('PERFORMING BACKPROPAGATION')
        direction = 1
        n = np.abs(v)
        for edge in path:
            edge.N += n
            edge.W += v * direction
            direction *= -1
            edge.Q = edge.W / edge.N
            lg.logger_mcts.info('Act = {}, N = {}, W = {}, Q = {}'.format(edge.action, edge.N, edge.W, edge.Q))
