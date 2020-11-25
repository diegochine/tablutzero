from game import Game
from memory import Memory
from player import Player
import config as cfg
import loggers as lg


def self_play(p1: Player, p2: Player, memory: Memory):
    """ play one match of self play, saving the (partial) results in memory"""
    game = Game()
    p1.reset()
    p2.reset()

    while not game.current_state.is_terminal:
        print('CURRENT TURN:', endgame_map[game.current_player])
        print(game.current_state.board, '\n')
        if game.current_player == 1:
            turn = 'WHITE'
            act, pi = p1.act(game.current_state)
        else:
            turn = 'BLACK'
            act, pi = p2.act(game.current_state)
        lg.logger_pure.info('{} TURN, ACTION: {}'.format(turn, act))
        memory.commit_stmemory(game.current_state, pi, None)
        game.execute(act)

    if game.current_state.value == 0:  # it's a draw
        lg.logger_pure.info("IT'S A DRAW")
        memory.commit_ltmemory(0)
    else:  # the player of this turn has lost
        memory.commit_ltmemory(-game.current_state.turn)
        lg.logger_pure.info('WINNER OF THIS EPISODE: {}'.format(endgame_map[-game.current_state.turn]))


if __name__ == "__main__":
    # SETUP GAME
    endgame_map = {0: 'DRAW', 1: 'WHITE', -1: 'BLACK'}

    memory = Memory(cfg.MEMORY_SIZE)

    # CREATE PLAYERS
    white = Player(color='WHITE', name='dc', timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)
    black = Player(color='BLACK', name='pd', timeout=cfg.TIMEOUT, simulations=cfg.MCTS_SIMULATIONS)

    for episode in range(cfg.EPISODES):
        print('EPISODE {:0>3d}/{:0>3d}'.format(episode, cfg.EPISODES))
        lg.logger_pure.info('EPISODE {:0>3d}/{:0>3d}'.format(episode, cfg.EPISODES))
        self_play(white, black, memory)
        memory.save(episode)
        memory.clear_ltmemory()