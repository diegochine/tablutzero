from collections import deque

import pytablut.config as cfg
import pytablut.loggers as lg

logger = lg.logger_memory
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
logger.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')


class Memory():

    def __init__(self, size=cfg.MEMORY_SIZE):
        self.MEMORY_SIZE = cfg.MEMORY_SIZE
        self.ltmemory = deque(maxlen=size)
        self.stmemory = deque(maxlen=size)

    def commit_stmemory(self, state, pi, value):
        """

        :param state: State object
        :param pi: search probabilities
        :param value: value of the state
        :return:
        """
        logger.info('ADDING NEW STATE')
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
        logger.info('CLEANING MEMORY')
        self.stmemory.clear()

    def save(self):
        pass