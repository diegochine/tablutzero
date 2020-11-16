import numpy as np
from networkx import DiGraph

from src.pytablut.State import State


class MonteCarloTreeSearch:

    def __init__(self, win_score=3, loss_score=0):
        self.citadels = {(0, 3), (0, 4), (0, 5), (1, 4),
                         (3, 0), (3, 8), (4, 0), (4, 1),
                         (4, 4),  # throne
                         (4, 7), (4, 8), (5, 0), (5, 8),
                         (7, 4), (8, 3), (8, 4), (8, 5)}
        self.escapes = {(0, 1), (0, 2), (0, 6), (0, 7),
                        (1, 0), (2, 0), (6, 0), (7, 0),
                        (8, 1), (8, 2), (8, 6), (8, 7),
                        (1, 8), (2, 8), (6, 8), (7, 8)}
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
            return king in self.escapes
        else:
            return 'KING' not in board

    def _utility(self, state, player):
        if state.turn == player + 'WIN':
            return self.win_score
        else:
            return self.loss_score

    def compute_move(self, current_state):
        if current_state not in self.search_space:
            self.search_space.add_node(current_state, score=0, visits=0, checkers=current_state.get_checkers())
        if len(self.search_space.adj[current_state]) == 0:
            # if node has no children, we expand it
            for action in self._actions(current_state):
                next_state = self._transition_function(current_state, action)
                self.search_space.add_node(next_state, score=0, visits=0, checkers=next_state.get_checkers())
                self.search_space.add_edge(current_state, next_state)

        # Selection Selecting good child nodes, starting from the root node R, that represent states leading to
        # better overall outcome (win).
        while True:  # TODO implement timeout
            leaf = self._select_node(current_state)
            # Expansion If L is a not a terminal node (i.e. it does not end the game), then create one or more
            # child nodes and select one (C).
            next_state = self._expand(leaf)
            # Simulation (rollout)
            # Run a simulated playout from C until a result is achieved.
            score = self._simulate_playout(next_state)
            # Backpropagation
            self._backpropagate(next_state, score)

    def _select_node(self, state):
        # UCB1: vi + 2 sqrt(ln(N)/ni)
        # Vi is the average reward/value of all nodes beneath this node
        # N is the number of times the parent node has been visited, and
        # ni is the number of times the child node i has been visited
        pass

    def _simulate_playout(self, state):
        return 0

    def _expand(self, state):
        a = self._actions(state)

    def _backpropagate(self, next_state, score):
        pass
