from collections import deque

import pytablut.config as cfg
import pytablut.loggers as lg

logger = lg.logger_memory
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
logger.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
logger.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')


class Memory():

    def __init__(self, MEMORY_SIZE=cfg.MEMORY_SIZE):
        self.MEMORY_SIZE = cfg.MEMORY_SIZE
        self.ltmemory = deque(maxlen=MEMORY_SIZE)
        self.stmemory = deque(maxlen=MEMORY_SIZE)

    def commit_stmemory(self, state, pi, value):
        logger.info('ADDING NEW STATE')
        self.stmemory.append({'state': state,
                              'id': state.id,
                              'pi': pi,
                              'value': value,
                              'playerTurn': state.turn})

    def commit_ltmemory(self, value):
        for i in self.stmemory:
            i['value'] = value
            self.ltmemory.append(i)
        self.clear_stmemory()

    def clear_stmemory(self):
        logger.info('CLEANING MEMORY')
        self.stmemory.clear()

    def save(self):
        pass