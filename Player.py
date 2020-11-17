import json
from socket import socket, AF_INET, SOCK_STREAM

import numpy as np

from src.pytablut.game import State
from src.pytablut.MCTS import MCTS


class Player:
    
    WHITE_PORT = 5800
    BLACK_PORT = 5801
    
    def __init__(self, color, name, timeout=60, ip_address='localhost', algo='mcts'):
        """
        :param color: color of the player, either BLACK or WHITE
        :param name: name of the player
        :param timeout: timeout in seconds for each move computation
        :param ip_address: string for the ip address
        """
        self.name = name
        self.color = color.upper()
        if self.color not in ('BLACK', 'WHITE'):
            raise ValueError('wrong color, must either BLACK or WHITE ')
        self.timeout = timeout
        if self.timeout <= 0:
            raise ValueError('timeout must be >0')
        if algo == 'mcts':
            self.algo = MCTS(self.color)
        else:
            raise ValueError('wrong algo parameter')
        self.board = None
        self.game_over = False
        self.turn = None
        self.ip_address = ip_address
        self.sock = self.__connect()

    def __json_to_state(self, state):
        state = State(board=np.array(state['board']),
                      turn=state['turn'])
        return state

    def __check_checkers(self):
        """checks if opponent move ate any checkers,
        and eventually removes them"""
        for (x, y) in list(self.checkers):
            if self.board[x, y] == 'EMPTY':
                self.checkers.remove((x, y))

    def __connect(self):
        """ performs TCP socket connection with the server"""
        self.sock = socket(AF_INET, SOCK_STREAM)
        if self.color == 'WHITE':
            conn = (self.ip_address, self.WHITE_PORT)
        elif self.color == 'BLACK':
            conn = (self.ip_address, self.BLACK_PORT)
        else:
            raise ValueError('color must be either white or black')
        self.sock.connect(conn)
        return self.sock

    def _coord_to_cell(self, coord):
        """
        performs conversion between
        :param coord: (x, y) int indexes of numpy matrix
        :return: a str "CR", where C is the column in [a-i] and R is the row in [1-9]
        """
        return ''.join([chr(coord[1]+97), str(coord[0]+1)])

    def __compute_move(self, state):
        """ computes moves based on current state and algo"""
        best_move = self.algo.compute_move(state)
        return best_move

    def declare_name(self):
        """declares name to the server"""
        print('Hello, my name is ' + self.name)
        self.write(self.name)
        print('Name declared')
        
    def write(self, action):
        """ writes a message to the server """
        tmp = json.dumps(action) + '\r\n'
        msg = bytearray(4 + len(tmp))
        msg[:4] = len(tmp).to_bytes(4, 'big')
        msg[4:] = tmp.encode()
        self.sock.sendall(msg)
    
    def read(self):
        """ reads a message from the server """
        msg = b''
        msg += self.sock.recv(1024)
        while len(msg) < 1 or chr(msg[-1]) != '}':
            tmp = self.sock.recv(1024)
            if not tmp:
                break
            msg += tmp
            if msg.find(ord('{'), 5) != -1:
                msg = msg[msg.find(ord('{'), 5) - 4:]
        msg = msg[4:].decode()
        json_state = json.loads(msg)
        state = self.__json_to_state(json_state)
        if state.turn not in ('BLACK', 'WHITE'):
            self.game_over = True
        return state

    def execute_move(self, move):
        act = {'from': self._coord_to_cell(move[0]),
               'to': self._coord_to_cell(move[1]),
               'turn': self.color}
        print(act)
        self.write(act)

    def play(self):
        self.declare_name()
        while not self.game_over:
            state = self.read()
            #self.__check_checkers()
            print('state received')
            if state.turn == self.color:
                print('my turn')
                move = self.__compute_move(state)
                print('move computed')
                self.execute_move(move)
                print('move executed')
            else:
                print('other player\'s turn')
        self.endgame()

    def endgame(self):
        """it's over mate"""
        if self.turn == 'DRAW':
            print("it\'s a draw")
        elif self.turn.startswith(self.color):
            print('i won!')
        else:
            print('i lost')
