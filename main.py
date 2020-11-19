import argparse

from pytablut.player import Player
from pytablut.game import Game

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to launch players.')
    parser.add_argument('timeout', type=int,
                        help='timeout in seconds')
    parser.add_argument('-s', '--simulations', type=int,
                        help='number of monte carlo simulations')
    parser.add_argument('-n', '--name', type=str, default='dioboia',
                        help='name of the player')

    args = parser.parse_args()
    white = Player(color='WHITE',
                   name='dc',
                   timeout=args.timeout,
                   simulations=args.simulations)
    black = Player(color='BLACK',
                   name='pd',
                   timeout=args.timeout,
                   simulations=args.simulations)
    game = Game()
    endgame_map = {0: 'DRAW', 1: 'WHITE', -1: 'BLACK'}

    while not game.current_state.is_terminal:
        print(game.current_state.board)
        if game.current_player == 1:
            print('WHITE')
            act = white.act(game.current_state)
        else:
            print('BLACK')
            act = black.act(game.current_state)
        print('move:', act)
        game.execute(act)

    print('Winner of the game is', endgame_map[game.current_state.value])
