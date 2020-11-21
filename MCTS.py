import numpy as np

from pytablut.game import State
import pytablut.loggers as lg
import pytablut.config as cfg

logger = lg.logger_mcts


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
        self.N = 0  # number of times action has been taken from initial state
        self.W = 0  # total value of next state
        self.Q = 0  # mean value of next state
        self.P = p  # prior probability of selecting this action


class MCTS:

    def __init__(self, player, root: Node, c_puct: float = cfg.CPUCT):
        self.player = player
        self.root: Node = root
        self.tree = {root.id: root}
        self.c_puct = c_puct
        self.expand_leaf(self.root)

    def change_root(self, state: State) -> None:
        self.root = self.tree[state.id]
        if self.root.is_leaf():
            self.expand_leaf(self.root)

    def add_node(self, node: Node):
        self.tree[node.id] = node

    def select_leaf(self) -> (Node, list):
        c = np.sqrt(2)
        node = self.root
        path = []

        while not node.is_leaf():
            max_uct = -np.inf
            N = np.sum([edge.N for edge in node.edges])
            simulation_edge = None

            for edge in node.edges:
                if edge.N == 0:
                    uct = np.inf
                else:
                    uct = edge.Q + np.sqrt(c) * np.sqrt(np.log(N) / edge.N)
                if uct > max_uct:
                    max_uct = uct
                    simulation_edge = edge
            next_state = node.state.transition_function(simulation_edge.action)
            node = simulation_edge.out_node
            path.append(simulation_edge)

        return node, path

    def expand_leaf(self, leaf: Node, p):
        for action in leaf.state.actions:
            next_state = leaf.state.transition_function(action)
            if next_state.id not in self.tree:
                new_leaf = Node(next_state)
                self.add_node(new_leaf)
            else:
                new_leaf = self.tree[next_state.id]
            new_edge = Edge(leaf, new_leaf, action, p[action])
            leaf.edges.append(new_edge)

    def random_playout(self, leaf: Node):
        logger.info('PERFORMING RANDOM PLAYOUT')
        state = leaf.state
        while not state.is_terminal:
            acts = state.actions
            # FIXME sometimes during random playout we have 0 actions and it crashes (low >= high)
            rnd_a = acts[np.random.randint(0, len(acts))]
            state = state.transition_function(rnd_a)
        return state.value

    def backpropagation(self, v: float, path: list):
        logger.info('PERFORMING BACKPROPAGATION')
        direction = -1
        for edge in path:
            edge.N += 1
            edge.W += v * direction
            direction *= -1
            edge.Q = edge.W / edge.N

    def choose_action(self) -> tuple:
        logger.info()
        best_n = np.argmax([edge.N for edge in self.root.edges])
        return self.root.edges[best_n].action
