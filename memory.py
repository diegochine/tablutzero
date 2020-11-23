from collections import deque
import pickle

import config as cfg
import loggers as lg


lg.logger_memory.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_memory.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_memory.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')


class Memory:

    def __init__(self, size=cfg.MEMORY_SIZE):
        self.MEMORY_SIZE = cfg.MEMORY_SIZE
        self.ltmemory = deque(maxlen=size)
        self.stmemory = deque(maxlen=size)

    def __len__(self):
        return len(self.ltmemory)

    def commit_stmemory(self, state, pi, value):
        """

        :param state: State object
        :param pi: search probabilities
        :param value: value of the state
        :return:
        """
        lg.logger_memory.info('ADDING NEW STATE')
        self.stmemory.append({'state': state,
                              'id': state.id,
                              'pi': pi,
                              'value': value,
                              'playerTurn': state.turn})

    def commit_ltmemory(self, winner):
        for i in self.stmemory:
            if i['playerTurn'] == winner:
                i['value'] = winner
            else:
                i['value'] = -winner
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        lg.logger_memory.info('CLEANING SHORT TERM MEMORY')
        self.stmemory.clear()

    def save(self, version):
        pickle.dump(self.ltmemory, open('memories/mem{}.pkl'.format(version), 'wb'))

    def clear_ltmemory(self):
        lg.logger_memory.info('CLEANING LONG TERM MEMORY')
        self.ltmemory.clear()
