import pytablut.config as cfg
import pytablut.loggers as lg
from pytablut.memory import Memory
from pytablut.neuralnet import ResidualNN
from pytablut.player import Player
from pytablut.game import Game

logger = lg.logger_train
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
logger.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

# CREATE MEMORY STORAGE
memory = Memory(cfg.MEMORY_SIZE)

# CREATE (AND EVENTUALLY LOAD) NETWORKS
general_nn = ResidualNN()



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
