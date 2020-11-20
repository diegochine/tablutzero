from collections import deque

import pytablut.config as cfg


class Memory(deque):

    def __init__(self, size=cfg.MEMORY_SIZE):
        super().__init__(maxlen=size)

    def commit_stmemory(self, state, pi, value):
        self.append({
            'board': state.board
            , 'id': state.id
            , 'pi': pi
            , 'value': value
            , 'playerTurn': state.turn
        })

    def clear_stmemory(self):
        super().clear()