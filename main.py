import argparse

from src.pytablut.player import Player
from src.pytablut.game import Game

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to launch players.')
    parser.add_argument('--timeout', type=int,
                        help='timeout in seconds')
    parser.add_argument('--ip', type=str,
                        help='server ip address')
    parser.add_argument('-n', '--name', type=str, default='dioboia',
                        help='name of the player')

    args = parser.parse_args()
    white = Player(color='WHITE',
                   name='dc',
                   timeout=60)
    black = Player(color='BLACK',
                   name='pd',
                   timeout=60)
    game = Game()

    for i in range(10):
        print(game.current_state.board)
        if i % 2 == 0:
            print('WHITE')
            act = white.act(game.current_state)
        else:
            print('BLACK')
            act = black.act(game.current_state)
        game.execute(act)
