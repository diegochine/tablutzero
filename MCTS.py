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
        self.tree = {root.id: root}
        self.c_puct = c_puct
        self.new_root(self.root.state)

    def new_root(self, state: State) -> None:
        if self.root.state.id != state.id:
            tmp = self.root
            self.root = self.tree[state.id]
            for edge in tmp.edges:
                self._delete_subtree(edge)
            del tmp
        if self.root.is_leaf():
            self.expand_leaf(self.root)

    def add_node(self, node: Node):
        self.tree[node.id] = node

    def ucb1(self, total_visit, node_win_score, node_visit):
        # UCB1: vi + 2 sqrt(ln(N)/ni)
        # Vi is the average reward/value of all nodes beneath this node
        # N is the number of times the parent node has been visited, and
        # ni is the number of times the child node i has been visited
        if node_visit == 0:
            return np.inf
        else:
            return (node_win_score / node_visit) + self.c_puct * np.sqrt(np.log(total_visit) / node_visit)

    def select_leaf(self) -> (Node, list):
        lg.logger_mcts.info('SELECTING LEAF')
        node = self.root
        path = []

        while not node.is_leaf():
            parent_visit = np.sum([edge.N for edge in node.edges])
            max_uct = - np.inf
            best = None
            for e in node.edges:
                uct = self.ucb1(parent_visit, e.W, e.N)
                if uct > max_uct:
                    max_uct = uct
                    best = e
            node = best.out_node
            path.append(best)

        return node, path

    def expand_leaf(self, leaf: Node):
        lg.logger_mcts.info('EXPANDING LEAF WITH ID {}'.format(leaf.id))
        for action in leaf.state.get_actions():
            next_state = leaf.state.transition_function(action)
            if next_state.id not in self.tree:
                new_leaf = Node(next_state)
                self.add_node(new_leaf)
                new_edge = Edge(leaf, new_leaf, action)
                leaf.edges.append(new_edge)

    def random_playout(self, leaf: Node):
        lg.logger_mcts.info('PERFORMING RANDOM PLAYOUT')
        state = leaf.state
        visited = {state.id}
        while not state.is_terminal:
            acts = state.get_actions()
            # FIXME sometimes during random playout we have 0 actions and it crashes (low >= high) TO BE VERIFIED
            rnd_a = acts[np.random.randint(0, len(acts))]
            state = state.transition_function(rnd_a)
            if state in visited:
                return 0 # it' a Draw
            visited.add(state)
        return state.value

    def backpropagation(self, v: float, path: list):
        lg.logger_mcts.info('PERFORMING BACKPROPAGATION WITH v = {:.2f}'.format(v[0]))
        direction = -1
        for edge in path:
            edge.N += 1
            edge.W += v * direction
            direction *= -1
            edge.Q = edge.W / edge.N
            # lg.logger_mcts.info('Act = {}, N = {}, W = {}, Q = {}'.format(edge.action, edge.N, edge.W, edge.Q))

    def _delete_subtree(self, edge):
        node = edge.out_node
        if self.root is None or node != self.root:
            del edge.out_node
            del edge.in_node
            del edge
            for out_edge in node.edges:
                self._delete_subtree(out_edge)
            del node.edges
        try:
            del self.tree[node.id]
        except KeyError:
            pass
        finally:
            del node

    def delete_tree(self):
        tmp = self.root
        self.root = None
        for edge in tmp.edges:
            self._delete_subtree(edge)
        del tmp.edges
        del tmp
