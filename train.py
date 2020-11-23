import config as cfg
import loggers as lg
from game import Game
from memory import Memory
from neuralnet import ResidualNN
from player import Player

lg.logger_train.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_train.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_train.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

# SETUP GAME
endgame_map = {0: 'DRAW', 1: 'WHITE', -1: 'BLACK'}

# CREATE MEMORY STORAGE
memory = Memory(cfg.MEMORY_SIZE)

# CREATE (AND EVENTUALLY LOAD) NETWORKS
general_nn = ResidualNN()
prototype_nn = ResidualNN()
lg.logger_train.info('LOADED NETWORK')

# CREATE PLAYERS
white = Player(color='WHITE', name='dc', nnet=general_nn,
               timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)
black = Player(color='BLACK', name='pd', nnet=general_nn,
               timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)

# START!
lg.logger_train.info('PLAYERS READY, STARTING MAIN LOOP')
for version in range(cfg.TOTAL_ITERATIONS):
    lg.logger_train.info('ITERATION NUMBER {:0>3d}/{:0>3d}'.format(version, cfg.TOTAL_ITERATIONS))
    lg.logger_train.info('SELF PLAYING FOR {:d} EPISODES'.format(cfg.EPISODES))

    black.brain.set_weights(white.brain.get_weights())

    for episode in range(cfg.EPISODES):
        lg.logger_train.info('EPISODE {:0>3d}/{:0>3d}'.format(episode, cfg.EPISODES))
        game = Game()
        white.reset()
        black.reset()

        while not game.current_state.is_terminal:
            print('CURRENT TURN:', endgame_map[game.current_player])
            print(game.current_state.board, '\n')
            if game.current_player == 1:
                turn = 'WHITE'
                act, pi = white.act(game.current_state)
            else:
                turn = 'BLACK'
                act, pi = black.act(game.current_state)
            lg.logger_train.info('{} TURN, ACTION: {}'.format(turn, act))
            memory.commit_stmemory(game.current_state, pi, None)
            game.execute(act)

        if game.current_state.value == 0:  # it's a draw
            lg.logger_train.info("IT'S A DRAW")
            memory.commit_ltmemory(0)
        else:  # the player of this turn has lost
            memory.commit_ltmemory(-game.current_state.turn)
            lg.logger_train.info('WINNER OF THIS EPISODE: {}'.format(endgame_map[-game.current_state.turn]))
        memory.save('v{:3d}ep{:3d}'.format(version, episode))
        memory.clear_ltmemory()

    lg.logger_train.info('RETRAINING NETWORK')
    if len(memory) >= cfg.MEMORY_SIZE:
        white.replay(memory.ltmemory)
        white.brain.save('general', version)
        memory.clear_stmemory()
        memory.clear_ltmemory()

    # TODO evaluate network
