import os
from collections import deque
import pickle

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
        :return:
        """
        lg.logger_memory.info('ADDING STATE WITH ID {}'.format(state.id))
        self.stmemory.append({'state': state,
                              'id': state.id,
                              'value': None,
                              'turn': state.turn})

    def commit_ltmemory(self, winner):
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

    def save(self, version):
        pickle.dump(self.ltmemory, open('memories/mem{}.pkl'.format(version), 'wb'))

    def clear_ltmemory(self):
        lg.logger_memory.info('CLEANING LONG TERM MEMORY')
        self.ltmemory.clear()
