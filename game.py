import numpy as np

MAP = {'EMPTY': 0, 'BLACK': -1, 'WHITE': 1, 'KING': 2}


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
        self.past_states = set()
        self.past_states.add(self.current_state.id)

    def execute(self, action):
        self.current_state = self.current_state.transition_function(action)
        if self.current_state.id in self.past_states:
            self.current_state.is_terminal = True
            self.current_state.value = 0
        else:
            self.past_states.add(self.current_state.id)
        self.current_player = -self.current_player


class State:
    def __init__(self, board, turn):
        """
        representation of the state:
        :param board: 9x9 matrix, filled with values according to map
        :param turn: current player, according to map
        the map is: 'EMPTY': 0, 'BLACK': -1, 'WHITE': 1, 'KING': 2
        """
        self.board: np.ndarray = board
        self.turn: int = turn
        self.id: int = self.__hash__()
        self.value: int = 0
        self.checkers: set = self._get_checkers()
        self.actions: list = []
        self.is_terminal: bool = self._terminal_test()

    def __hash__(self):
        return hash((tuple(tuple(row) for row in self.board), self.turn))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return str(self.board)

    def _get_checkers(self) -> set:
        """ return positions of checkers that can be moved in this state """
        checkers = set(tuple(x) for x in np.argwhere(self.board == self.turn))
        if self.turn == 1 and 2 in self.board:
            checkers.add(tuple(np.argwhere(self.board == 2)[0]))
        return checkers

    def _terminal_test(self) -> bool:
        king = tuple(np.argwhere(self.board == 2).flatten())
        white_win = king in Game.escapes or -1 not in self.board
        black_win = 2 not in self.board
        if (white_win or black_win):  # or not self.actions:
            # either the current player has lost or he cannot move (so he lost)
            self.value = -1
            return True
        else:
            return False

    def __same_citadel_area(self, start: tuple, end: tuple) -> bool:
        """
        :returns True if both start and end are citadel cells and they are in the same group of citadels;
        False otherwise"""
        if start in Game.citadels and end in Game.citadels:
            return np.abs(start[0] - end[0]) + np.abs(start[1] - end[1]) <= 2
        else:
            return False

    def get_actions(self) -> list:
        if not self.actions:
            for (x, y) in self.checkers:
                # try up
                newx = x - 1
                while newx >= 0 and self.board[newx, y] == 0 and \
                        ((newx, y) not in Game.citadels or self.__same_citadel_area((x, y), (newx, y))):
                    self.actions.append(((x, y), (newx, y)))
                    newx -= 1

                # try down
                newx = x + 1
                while newx <= 8 and self.board[newx, y] == 0 and \
                        ((newx, y) not in Game.citadels or self.__same_citadel_area((x, y), (newx, y))):
                    self.actions.append(((x, y), (newx, y)))
                    newx += 1

                # try left
                newy = y - 1
                while newy >= 0 and self.board[x, newy] == 0 and \
                        ((x, newy) not in Game.citadels or self.__same_citadel_area((x, y), (x, newy))):
                    self.actions.append(((x, y), (x, newy)))
                    newy -= 1

                # try right
                newy = y + 1
                while newy <= 8 and self.board[x, newy] == 0 and \
                        ((x, newy) not in Game.citadels or self.__same_citadel_area((x, y), (x, newy))):
                    self.actions.append(((x, y), (x, newy)))
                    newy += 1

        return self.actions

    def transition_function(self, action: tuple):
        """
        Given an action, returns the state resulting from applying the action to this state
        :param action: tuple ((x_from, y_from), (x_to, y_to))
        :return: State object with updated board and turn
        """
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

    def _check_enemy_capture(self, board: np.ndarray, arrival: tuple, enemy: int):
        row_to, col_to = arrival
        if col_to < board.shape[1] - 2 and board[row_to, col_to + 1] == enemy and (row_to, col_to + 1) not in Game.citadels:
            # on the right there's an enemy
            over_the_enemy = (row_to, col_to + 2)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to, col_to + 1] = 0
        if col_to > 1 and board[row_to, col_to - 1] == enemy and (row_to, col_to - 1) not in Game.citadels:
            # on the left there's an enemy
            over_the_enemy = (row_to, col_to - 2)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to, col_to - 1] = 0
        if row_to > 1 and board[row_to - 1, col_to] == enemy and (row_to - 1, col_to) not in Game.citadels:
            # above there's an enemy
            over_the_enemy = (row_to - 2, col_to)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to - 1, col_to] = 0
        if row_to < board.shape[0] - 2 and board[row_to + 1, col_to] == enemy and (row_to + 1, col_to) not in Game.citadels:
            # below there's an enemy
            over_the_enemy = (row_to + 2, col_to)
            if board[over_the_enemy] == self.turn or over_the_enemy in Game.citadels:
                # remove the checker
                board[row_to + 1, col_to] = 0
        return board

    def convert_into_cnn(self) -> np.array:
        """
        Converts this state as input for the neural network
        :return: np.array of shape (9x9x4)
        """
        black = np.where(self.board == -1, 1, 0)
        white = np.where(self.board == 1, 1, 0)
        king = np.where(self.board == 2, 1, 0)
        if self.turn == -1:
            turn = np.zeros((9, 9), dtype=int)
        else:
            turn = np.ones((9, 9), dtype=int)
        return np.stack([black, white, king, turn], axis=2)
