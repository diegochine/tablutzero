import numpy as np


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

    s0 = np.array([[0, 0, 0, -1, -1, -1, 0, 0, 0],
                   [0, 0, 0, 0, -1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [-1, 0, 0, 0, 1, 0, 0, 0, -1],
                   [-1, -1, 1, 1, 2, 1, 1, -1, -1],
                   [-1, 0, 0, 0, 1, 0, 0, 0, -1],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, -1, 0, 0, 0, 0],
                   [0, 0, 0, -1, -1, -1, 0, 0, 0]])

    def __init__(self):
        self.current_player: int = 1
        self.current_state: State = State(board=self.s0, turn=1)

    def execute(self, action):
        self.current_state = self.current_state.transition_function(action)


class State:

    def __init__(self, board, turn):
        """
        representation of the state:
        :param board: 9x9 matrix, filled with values according to map
        :param turn: current player, according to map
        the map is: 'EMPTY': 0, 'BLACK': -1, 'WHITE': 1, 'KING': 2
        """
        self.board = board
        self.turn = turn
        self.id = self.__hash__()
        self.value = 0
        self.is_terminal = self._terminal_test()
        self.checkers = self._get_checkers()
        self.actions = self._get_actions()

    def __hash__(self):
        return hash((tuple(tuple(row) for row in self.board), self.turn))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.board)

    def _get_checkers(self):
        """ return positions of checkers that can be moved in this state """
        checkers = set(tuple(x) for x in np.argwhere(self.board == self.turn))
        if self.turn == 1 and 2 in self.board:
            checkers.add(tuple(np.argwhere(self.board == 2)[0]))
        return checkers

    def _terminal_test(self):
        king = tuple(np.argwhere(self.board == 2).flatten())
        white_win = king in Game.escapes or -1 not in self.board
        black_win = 2 not in self.board
        if white_win or black_win:
            # TODO need to check, black player needs to minimize whit values assigned like this?
            if white_win:
                self.value = 1
            else:
                self.value = -1
            return True
        else:
            return False

    def _get_actions(self, ):
        # TODO let black checkers go out of the citadels (can't go back)
        actions = []
        for (x, y) in self.checkers:
            offx = 1
            # try up
            while (x - offx) >= 0 and (x - offx, y) not in Game.citadels and self.board[x - offx, y] == 0:
                actions.append(((x, y), (x - offx, y)))
                offx += 1

            # try down
            offx = 1
            while (x + offx) <= 8 and (x + offx, y) not in Game.citadels and self.board[x + offx, y] == 0:
                actions.append(((x, y), (x + offx, y)))
                offx += 1

            # try left
            offy = 1
            while (y - offy) >= 0 and (x, y - offy) not in Game.citadels and self.board[x, y - offy] == 0:
                actions.append(((x, y), (x, y - offy)))
                offy += 1

            # try right
            offy = 1
            while (y + offy) <= 8 and (x, y + offy) not in Game.citadels and self.board[x, y + offy] == 0:
                actions.append(((x, y), (x, y + offy)))
                offy += 1

        return actions

    def transition_function(self, action):
        pos_start, pos_end = action
        board = self.board.copy()
        board[pos_start], board[pos_end] = board[pos_end], board[pos_start]
        # check if any enemy checkers got eaten
        board = self._check_enemy_capture(board, pos_end, -self.turn)
        if self.turn == -1:
            # checking capture of the king in black's turn
            if board[4, 4] == 2:
                # king in the throne
                if board[3, 4] == -1 and board[5, 4] == -1 and board[4, 5] == -1 and board[4, 3] == -1:
                    # king is surrounded, remove it
                    board[4, 4] = 0
            else:
                board = self._check_enemy_capture(board, pos_end, 2)

        return State(board=board, turn=-self.turn)

    def _check_enemy_capture(self, board, arrival, enemy):
        row_to, col_to = arrival
        if col_to < board.shape[1] - 2 and board[row_to, col_to + 1] == enemy:
            # on the right there's an enemy
            over_the_enemy = (row_to, col_to + 2)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to, col_to + 1] = 0
        if col_to > 1 and board[row_to, col_to - 1] == enemy:
            # on the left there's an enemy
            over_the_enemy = (row_to, col_to - 2)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to, col_to - 1] = 0
        if row_to > 1 and board[row_to - 1, col_to] == enemy:
            # above there's an enemy
            over_the_enemy = (row_to - 2, col_to)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to - 1, col_to] = 0
        if row_to < board.shape[0] - 2 and board[row_to + 1, col_to] == enemy:
            # below there's an enemy
            over_the_enemy = (row_to + 2, col_to)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to + 1, col_to] = 0
        return board
