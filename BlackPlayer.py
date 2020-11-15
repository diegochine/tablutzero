from src.pytablut.Player import Player


class BlackPlayer(Player):

    def __init__(self, name, timeout=60, ip_address='localhost', algo='mcts'):
        super().__init__('black', name, timeout, ip_address, algo)
        self.checkers = {(0, 3), (0, 4), (0, 5), (1, 4),
                         (3, 0), (3, 8), (4, 0), (4, 1),
                         (4, 7), (4, 8), (5, 0), (5, 8),
                         (7, 4), (8, 3), (8, 4), (8, 5)}

    def __get_moves(self):
        moves = []
        for (x, y) in self.checkers:
            offx = 1
            # try up
            while (x - offx) >= 0 and (x - offx, y) not in self.citadels and self.board[x - offx, y] == 'EMPTY':
                moves.append(((x, y), (x - offx, y)))
                offx += 1

            offx = 1
            while (x + offx) <= 8 and (x + offx, y) not in self.citadels and self.board[x + offx, y] == 'EMPTY':
                moves.append(((x, y), (x + offx, y)))
                offx += 1

            offy = 1
            while (y - offy) >= 0 and (x, y - offy) not in self.citadels and self.board[x, y - offy] == 'EMPTY':
                moves.append(((x, y), (x, y - offy)))
                offy += 1

            offy = 1
            while (y + offy) <= 8 and (x, y + offy) not in self.citadels and self.board[x, y + offy] == 'EMPTY':
                moves.append(((x, y), (x, y + offy)))
                offy += 1

        return moves

    def play(self):
        self.declare_name()
        while not self.game_over:
            self.read()
            self.__check_checkers()
            print('state received')
            if self.turn == 'BLACK':
                print('my turn')
                move = self.__compute_move()
                print('move computed')
                self.execute_move(move)
                print('move executed')
            elif self.turn == 'WHITE':
                print('other player\'s turn')
        self.endgame()
