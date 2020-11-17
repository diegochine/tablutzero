import numpy as np


class State:

    def __init__(self, board, turn):
        self.board = board
        self.turn = turn
        self.checkers = self._get_checkers()
        self.actions = self._get_actions()
        self.is_terminal = self._terminal_test()

    def __hash__(self):
        return hash((tuple(tuple(row) for row in self.board), self.turn))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.board)

    def _get_checkers(self):
        """ return positions of checkers that can be moved in this state """
        checkers = set(tuple(x) for x in np.argwhere(self.board == self.turn))
        if self.turn == 'WHITE':
            checkers.add(tuple(np.argwhere(self.board == 'KING')[0]))
        return checkers

    def _terminal_test(self):
        king = tuple(np.argwhere(self.board == 'KING').flatten())
        white_win = king in Game.escapes or 'BLACK' not in self.board
        black_win = 'KING' not in self.board
        return white_win or black_win

    def _get_actions(self, ):
        # TODO let black checkers go out of the citadels (can't go back)
        actions = []
        for (x, y) in self.checkers:
            offx = 1
            # try up
            while (x - offx) >= 0 and (x - offx, y) not in Game.citadels and self.board[x - offx, y] == 'EMPTY':
                actions.append(((x, y), (x - offx, y)))
                offx += 1

            # try down
            offx = 1
            while (x + offx) <= 8 and (x + offx, y) not in Game.citadels and self.board[x + offx, y] == 'EMPTY':
                actions.append(((x, y), (x + offx, y)))
                offx += 1

            # try left
            offy = 1
            while (y - offy) >= 0 and (x, y - offy) not in Game.citadels and self.board[x, y - offy] == 'EMPTY':
                actions.append(((x, y), (x, y - offy)))
                offy += 1

            # try right
            offy = 1
            while (y + offy) <= 8 and (x, y + offy) not in Game.citadels and self.board[x, y + offy] == 'EMPTY':
                actions.append(((x, y), (x, y + offy)))
                offy += 1

        return actions

    def transition_function(self, action):
        pos_start, pos_end = action
        board = self.board.copy()
        board[pos_start], board[pos_end] = board[pos_end], board[pos_start]
        # TODO check if checkers got eaten
        if self.turn == 'WHITE':
            next_turn = 'BLACK'
        else:
            next_turn = 'WHITE'
        return State(board=board, turn=next_turn)


class Game:
    citadels = {(0, 3), (0, 4), (0, 5), (1, 4),
                (3, 0), (3, 8), (4, 0), (4, 1),
                (4, 4),  # throne
                (4, 7), (4, 8), (5, 0), (5, 8),
                (7, 4), (8, 3), (8, 4), (8, 5)}
    escapes = {(0, 1), (0, 2), (0, 6), (0, 7),
               (1, 0), (2, 0), (6, 0), (7, 0),
               (8, 1), (8, 2), (8, 6), (8, 7),
               (1, 8), (2, 8), (6, 8), (7, 8)}

    s0 = State(board=np.array([[0, 0, 0, -1, -1, -1, 0, 0, 0],
                               [0, 0, 0, 0, -1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [-1, 0, 0, 0, 1, 0, 1, 0, 0],
                               [-1, -1, 1, 0, 1, 1, 0, -1, -1],
                               [-1, 0, 0, 0, 1, 0, 0, 0, -1],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, -1],
                               [0, 0, 0, -1, -1, 0, -1, 0, 0]]),
               turn=1)

    def __init__(self):
        self.current_player = 1
        self.current_state = self.s0

    def execute(self, action):
        pass
