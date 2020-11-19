import pytablut.config as cfg
from pytablut.player import Player
from pytablut.game import Game

if __name__ == '__main__':
    white = Player(color='WHITE',
                   name='dc',
                   timeout=cfg.TIMEOUT,
                   simulations=cfg.MCTS_SIMULATIONS)
    black = Player(color='BLACK',
                   name='pd',
                   timeout=cfg.TIMEOUT,
                   simulations=cfg.MCTS_SIMULATIONS)
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
