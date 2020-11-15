from src.pytablut.Player import Player


class WhitePlayer(Player):

    def __init__(self, name, timeout=60, ip_address='localhost', algo='mcts'):
        super().__init__('white', name, timeout, ip_address, algo)
        self.checkers = {(2, 4), (3, 4), (4, 2), (4, 3),
                         (4, 5), (4, 6), (5, 4), (6, 4),
                         (4, 4)}  # THE KING

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
            if self.turn == 'WHITE':
                print('my turn')
                move = self.__compute_move()
                print('move computed')
                self.execute_move(move)
                print('move executed')
            elif self.turn == 'BLACK':
                print('other player\'s turn')
        self.endgame()
