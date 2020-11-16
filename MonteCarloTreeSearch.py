import numpy as np
from networkx import DiGraph

from src.pytablut.State import State


class MonteCarloTreeSearch:

    def __init__(self, player, win_score=3, loss_score=0):
        self.citadels = {(0, 3), (0, 4), (0, 5), (1, 4),
                         (3, 0), (3, 8), (4, 0), (4, 1),
                         (4, 4),  # throne
                         (4, 7), (4, 8), (5, 0), (5, 8),
                         (7, 4), (8, 3), (8, 4), (8, 5)}
        self.escapes = {(0, 1), (0, 2), (0, 6), (0, 7),
                        (1, 0), (2, 0), (6, 0), (7, 0),
                        (8, 1), (8, 2), (8, 6), (8, 7),
                        (1, 8), (2, 8), (6, 8), (7, 8)}
        self.player = player
        self.win_score = win_score
        self.loss_score = loss_score
        self.search_space: DiGraph = DiGraph()
        s0 = State(board=np.array([['EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY'],
                                   ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                                   ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                                   ['BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY'],
                                   ['BLACK', 'BLACK', 'WHITE', 'EMPTY', 'KING', 'WHITE', 'EMPTY', 'BLACK', 'BLACK'],
                                   ['BLACK', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                                   ['EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY'],
                                   ['EMPTY', 'EMPTY', 'EMPTY', 'WHITE', 'EMPTY', 'EMPTY', 'EMPTY', 'EMPTY', 'BLACK'],
                                   ['EMPTY', 'EMPTY', 'EMPTY', 'BLACK', 'BLACK', 'EMPTY', 'BLACK', 'EMPTY', 'EMPTY']]),
                   turn='WHITE')
        self.search_space.add_node(s0)
        self.search_space.nodes[s0]['score'] = 0
        self.search_space.nodes[s0]['visits'] = 0
        self.search_space.nodes[s0]['checkers'] = {(2, 4), (3, 4), (4, 2), (4, 3),
                                                   (4, 5), (4, 6), (5, 4), (6, 4),
                                                   (4, 4)}

    def _actions(self, state):
        # TODO let black checkers go out of the citadels (can't go back)
        moves = []
        for (x, y) in self.search_space[state]['checkers']:
            offx = 1
            # try up
            while (x - offx) >= 0 and (x - offx, y) not in self.citadels and state.board[x - offx, y] == 'EMPTY':
                moves.append(((x, y), (x - offx, y)))
                offx += 1

            # try down
            offx = 1
            while (x + offx) <= 8 and (x + offx, y) not in self.citadels and state.board[x + offx, y] == 'EMPTY':
                moves.append(((x, y), (x + offx, y)))
                offx += 1

            # try left
            offy = 1
            while (y - offy) >= 0 and (x, y - offy) not in self.citadels and state.board[x, y - offy] == 'EMPTY':
                moves.append(((x, y), (x, y - offy)))
                offy += 1

            # try right
            offy = 1
            while (y + offy) <= 8 and (x, y + offy) not in self.citadels and state.board[x, y + offy] == 'EMPTY':
                moves.append(((x, y), (x, y + offy)))
                offy += 1

        return moves

    def _transition_function(self, state, action):
        pos_start, pos_end = action
        board = state.board.copy()
        board[pos_start], board[pos_end] = board[pos_end], board[pos_start]
        # TODO check if checkers got eaten
        if self._terminal_test(board, state.turn):
            next_turn = state.turn + 'WIN'
        elif state.turn == 'WHITE':
            next_turn = 'BLACK'
        else:
            next_turn = 'WHITE'
        return State(board=board, turn=next_turn)

    def _terminal_test(self, board, prec_move):
        if prec_move == 'WHITE':
            king = tuple(np.argwhere(board == 'KING').flatten())
            return king in self.escapes or 'BLACK' not in board
        else:
            return 'KING' not in board

    def _utility(self, state):
        if state.turn == self.player + 'WIN':
            return self.win_score
        else:
            return self.loss_score

    def compute_move(self, current_state):
        if current_state not in self.search_space:
            self.search_space.add_node(current_state, score=0, visits=0, checkers=current_state.get_checkers())
        if len(self.search_space.adj[current_state]) != len(self._actions(current_state)):
            # if node has no children, we expand it
            for action in self._actions(current_state):
                next_state = self._transition_function(current_state, action)
                self.search_space.add_node(next_state, score=0, visits=0, checkers=next_state.get_checkers())
                self.search_space.add_edge(current_state, next_state, action=action)

        # Selection Selecting good child nodes, starting from the root node R, that represent states leading to
        # better overall outcome (win).
        while True:  # TODO implement timeout
            leaf_state = self._select_node(current_state)
            # Expansion If L is a not a terminal node (i.e. it does not end the game), then create one or more
            # child nodes and select one (C).
            new_state = self._expand(leaf_state)
            # Simulation (rollout)
            # Run a simulated playout from C until a result is achieved.
            score = self._simulate_playout(new_state)
            # Backpropagation
            self._backpropagate(new_state, current_state, score)

    def _select_node(self, state):
        best_node = self.search_space.nodes[state]
        children = self.search_space.adj[best_node]
        while len(children) > 0:
            parent_visit = best_node['visits']
            max_uct = - np.inf
            # FIXME fare meglio
            for s in children:
                uct = self._ucb1(parent_visit, s['score'], s['visits'])
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
        actions = self._actions(state)
        a = actions[np.random.randint(0, len(actions))]
        child = self._transition_function(state, a)
        self.search_space.add_node(child, score=0, visits=0, checkers=child.get_checkers())
        self.search_space.add_edge(state, child, action=a)
        return child

    def _simulate_playout(self, state):
        while not state.turn.endswith('WIN'):
            acts = self._actions(state)
            a = acts[np.random.randint(0, len(acts))]
            state = self._transition_function(state, a)
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

