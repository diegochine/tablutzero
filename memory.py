from collections import deque

import pytablut.config as cfg


class Memory(deque):

    def __init__(self, size=cfg.MEMORY_SIZE):
        super().__init__(maxlen=size)
