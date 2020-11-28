import os
from collections import deque
import pickle
import numpy as np

import config as cfg
import loggers as lg
import game  # needed by pickle


def load_memories(path='./memories/'):
    lg.logger_memory.info("LOADING MEMORIES FROM STORAGE")
    try:
        with open(path + 'dataset.pkl', 'rb') as f:
            memories = pickle.load(f)
        lg.logger_memory.info("SIZE OF LOADED MEMORIES: {:0>5d}".format(len(memories)))
        return memories
    except FileNotFoundError:
        return None


def compact_memories(path='./memories/'):
    memories = pickle.load(open(path + 'dataset.pkl', 'rb'))
    for fname in os.listdir(path):
        if fname.endswith('pkl') and not fname.startswith('dataset'):
            with open(path + fname, 'rb') as f:
                memories.extend(pickle.load(f))
            os.remove(path + fname)
    pickle.dump(memories, open(path + 'dataset.pkl', 'wb'))


class Memory:

    def __init__(self, size=cfg.MEMORY_SIZE, ltmemory=None):
        self.MEMORY_SIZE = cfg.MEMORY_SIZE
        if ltmemory is not None:
            self.ltmemory = ltmemory
        else:
            self.ltmemory = deque(maxlen=size)
        self.stmemory = deque(maxlen=size)

    def __len__(self):
        return len(self.ltmemory)

    def commit_stmemory(self, state):
        """
        :param state: State object
        """
        # data augmentation exploiting symmetries
        for rot in range(4):
            rot_board = np.rot90(state.board, rot)
            new_state = game.State(board=rot_board, turn=state.turn)
            lg.logger_memory.info('ADDING STATE WITH ID {}'.format(new_state.id))
            self.stmemory.append({'state': new_state,
                                  'id': new_state.id,
                                  'value': None,
                                  'turn': new_state.turn})

    def commit_ltmemory(self, winner):
        lg.logger_memory.info('COMMITTING WINNER OF THIS EPISODE: {}'.format(winner))
        for mem in self.stmemory:
            if winner == 0:
                mem['value'] = 0
            elif mem['turn'] == winner:
                mem['value'] = 1
            else:
                mem['value'] = -1
            self.ltmemory.append(mem)
        self.clear_stmemory()

    def clear_stmemory(self):
        lg.logger_memory.info('CLEANING SHORT TERM MEMORY')
        self.stmemory.clear()

    def save(self, name):
        pickle.dump(self.ltmemory, open('memories/mem{}.pkl'.format(name), 'wb'))

    def clear_ltmemory(self):
        lg.logger_memory.info('CLEANING LONG TERM MEMORY')
        self.ltmemory.clear()
