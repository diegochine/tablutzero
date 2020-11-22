import config as cfg
import loggers as lg
from game import Game
from memory import Memory
from neuralnet import ResidualNN
from player import Player

logger = lg.logger_train
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
logger.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

# SETUP GAME
endgame_map = {0: 'DRAW', 1: 'WHITE', -1: 'BLACK'}

# CREATE MEMORY STORAGE
memory = Memory(cfg.MEMORY_SIZE)

# CREATE (AND EVENTUALLY LOAD) NETWORKS
general_nn = ResidualNN()
prototype_nn = ResidualNN()
logger.info('LOADED NETWORK')

# CREATE PLAYERS
white = Player(color='WHITE', name='dc', nnet=general_nn,
               timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)
black = Player(color='BLACK', name='pd', nnet=general_nn,
               timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)

# START!
logger.info('PLAYERS READY, STARTING MAIN LOOP')
for i in range(cfg.TOTAL_ITERATIONS):
    logger.info('ITERATION NUMBER {:0>3d}/{:0>3d}'.format(i, cfg.TOTAL_ITERATIONS))
    logger.info('SELF PLAYING FOR {:d} EPISODES'.format(cfg.EPISODES))

    # TODO make sure both players have the same weight

    for episode in range(cfg.EPISODES):
        logger.info('EPISODE {:0>3d}/{:0>3d}'.format(episode, cfg.EPISODES))
        game = Game()
        while not game.current_state.is_terminal:
            print(game.current_state.board, '\n')
            if game.current_player == 1:
                turn = 'WHITE'
                act, pi = white.act(game.current_state)
            else:
                turn = 'BLACK'
                act, pi = black.act(game.current_state)
            logger.info('{} TURN, ACTION: {}'.format(turn, act))
            memory.commit_stmemory(game.current_state, pi, None)
            game.execute(act)
        logger.info('WINNER OF THIS EPISODE: {}'.format(endgame_map[game.current_state.value]))
        memory.commit_ltmemory(-game.current_state.turn)

    logger.info('RETRAINING NETWORK')
    white.replay(memory)
    white.brain.save('general', i)

    # TODO evaluate network
