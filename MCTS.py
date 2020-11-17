import numpy as np

from src.pytablut.game import State


class Node:

    def __init__(self, state):
        self.state = state
        self.id = hash(state)
        self.edges = []

    def is_leaf(self):
        return len(self.edges) == 0


class Edge:

    def __init__(self, in_node, out_node, p, action):
        self.in_node = in_node
        self.out_node = out_node
        self.action = action
        self.stats = {'N': 0, 'W': 0,
                      'Q': 0, 'P': p}


class MCTS:

    def __init__(self, player, root):
        self.player = player
        self.root = root
        self.tree = {root.id: root}

    def compute_move(self, current_state: State):
        if current_state not in self.search_space:
            self.search_space.add_node(current_state, score=0, visits=0, checkers=current_state.get_checkers())
        if len(self.search_space.adj[current_state]) != 0:
            # if node has no children, we expand it
            for action in current_state.actions:
                next_state = self._transition_function(current_state, action)
                self.search_space.add_node(next_state, score=0, visits=0, checkers=next_state.get_checkers())
                self.search_space.add_edge(current_state, next_state, action=action)

        # Selection Selecting good child nodes, starting from the root node R, that represent states leading to
        # better overall outcome (win).
        print('starting playouts')
        for i in range(10**2):  # TODO implement timeout
            print(i)
            leaf_state = self._select_node(current_state)
            # Expansion If L is a not a terminal node (i.e. it does not end the game), then create one or more
            # child nodes and select one (C).
            new_state = self._expand(leaf_state)
            # Simulation (rollout)
            # Run a simulated playout from C until a result is achieved.
            score = self._simulate_playout(new_state)
            # Backpropagation
            self._backpropagate(new_state, current_state, score)
        print('playouts over')
        best_move = None
        best_visits = -1
        for succ in self.search_space.successors(current_state):
            if self.search_space.nodes[succ]['visits'] > best_visits:
                best_visits = self.search_space.nodes[succ]['visits']
                best_move = self.search_space.edges[current_state, succ]['action']
        return best_move

    def _select_node(self, state):
        best_node = state
        children = list(self.search_space.successors(best_node))
        # FIXME loop nel grafo
        while len(children) > 0:
            parent_visit = self.search_space.nodes[best_node]['visits']
            max_uct = - np.inf
            # FIXME fare meglio
            for s in children:
                uct = self._ucb1(parent_visit,
                                 self.search_space.nodes[s]['score'],
                                 self.search_space.nodes[s]['visits'])
                if uct > max_uct:
                    max_uct = uct
                    best_node = s

            children = self.search_space.adj[best_node]

        return best_node

    def _ucb1(self, total_visit, node_win_score, node_visit):
        # UCB1: vi + 2 sqrt(ln(N)/ni)
        # Vi is the average reward/value of all nodes beneath this node
        # N is the number of times the parent node has been visited, and
        # ni is the number of times the child node i has been visited
        if node_visit == 0:
            return np.inf
        else:
            return (node_win_score / node_visit) + 2 * np.sqrt(np.log(total_visit) / node_visit)

    def _expand(self, state):
        if state.actions:
            a = state.actions[np.random.randint(0, len(state.actions))]
            child = state.transition_function(a)
            self.search_space.add_node(child, score=0, visits=0, checkers=child.get_checkers())
            self.search_space.add_edge(state, child, action=a)
            return child
        else:
            return state

    def _simulate_playout(self, state):
        while not state.turn.endswith('WIN'):
            acts = state.actions
            a = acts[np.random.randint(0, len(acts))]
            state = state.transition_function(a)
        return self._utility(state)

    def _backpropagate(self, next_state, initial_state, score):
        pred = next_state
        while pred != initial_state:
            self.search_space.nodes[pred]['score'] += score
            self.search_space.nodes[pred]['visits'] += 1
            # FIXME farlo per tutti i "padri"?
            pred = list(self.search_space.predecessors(pred))[0]
        self.search_space.nodes[initial_state]['score'] += score
        self.search_space.nodes[initial_state]['visits'] += 1

