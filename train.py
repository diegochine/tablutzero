import config as cfg
import loggers as lg
from game import Game
from memory import Memory, load_memories, compact_memories
from neuralnet import ResidualNN
from player import Player
from utils import Timeit

lg.logger_train.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_train.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_train.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')


@Timeit(logger=lg.logger_train)
def self_play(p1: Player, p2: Player, memory: Memory):
    """ play one match of self play, saving the (partial) results in memory"""
    game = Game()
    p1.reset()
    p2.reset()

    while not game.current_state.is_terminal:
        print(game.current_state.board)
        if game.current_player == 1:
            turn = 'WHITE'
            act = p1.act(game.current_state)
        else:
            turn = 'BLACK'
            act = p2.act(game.current_state)
        lg.logger_train.info('{} TURN, ACTION: {}'.format(turn, act))
        print('{} TURN, ACTION: {}\n'.format(turn, act))
        memory.commit_stmemory(game.current_state)
        game.execute(act)

    if game.current_state.value == 0:  # it's a draw
        lg.logger_train.info("IT'S A DRAW")
        memory.commit_ltmemory(0)
    else:  # the player of this turn has lost
        memory.commit_ltmemory(-game.current_state.turn)
        lg.logger_train.info('WINNER OF THIS EPISODE: {}'.format(endgame_map[-game.current_state.turn]))


if __name__ == "__main__":
    # SETUP GAME
    endgame_map = {0: 'DRAW', 1: 'WHITE', -1: 'BLACK'}

    # LOAD MEMORY STORAGE
    ltmemory = load_memories()
    memory = Memory(cfg.MEMORY_SIZE, ltmemory)

    # CREATE (AND EVENTUALLY LOAD) NETWORKS
    lg.logger_train.info('LOADED NETWORK')

    # CREATE PLAYERS
    white = Player(color='WHITE', name='dc', nnet_ver=cfg.CURRENT_VERSION,
                   timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)
    black = Player(color='BLACK', name='pd', nnet_ver=cfg.CURRENT_VERSION,
                   timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)

    # START
    lg.logger_train.info('PLAYERS READY, STARTING MAIN LOOP')
    for version in range(cfg.TOTAL_ITERATIONS):
        lg.logger_train.info('ITERATION NUMBER {:0>3d}/{:0>3d}'.format(version, cfg.TOTAL_ITERATIONS))
        lg.logger_train.info('SELF PLAYING FOR {:d} EPISODES'.format(cfg.EPISODES))

        # make sure both players use the same network
        black.brain.set_weights(white.brain.get_weights())

        for episode in range(cfg.EPISODES):
            lg.logger_train.info('EPISODE {:0>3d}/{:0>3d}'.format(episode, cfg.EPISODES))
            self_play(white, black, memory)
            memory.save('ep{:0>4d}'.format(episode))
            memory.clear_ltmemory()

        compact_memories()
        ltmemory = load_memories()
        # memory = Memory(cfg.MEMORY_SIZE, ltmemory)

        lg.logger_train.info('RETRAINING NETWORK')
        white.replay(ltmemory)
        white.brain.save(version)

        # TODO evaluate network
