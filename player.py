from pytablut.MCTS import MCTS, Node
from pytablut.game import MAP


class Player:

    def __init__(self, color, name, timeout=60, simulations=500):
        """
        :param color: color of the player, either BLACK or WHITE
        :param name: name of the player
        :param timeout: timeout in seconds for each move computation
        """
        self.name = name
        if color not in ('BLACK', 'WHITE'):
            raise ValueError('wrong color, must either BLACK or WHITE ')
        self.color = MAP[color]
        self.timeout = timeout
        if self.timeout <= 0:
            raise ValueError('timeout must be >0')
        self.mcts = None
        self.simulations = simulations
        self.game_over = False

    def build_mcts(self, state):
        self.mcts = MCTS(self.color, Node(state))

    def act(self, state):
        """ computes best action based on given state"""
        if self.mcts is None or hash(state) not in self.mcts.tree:
            self.build_mcts(state)
        else:
            self.mcts.change_root(state)

        # time to roll
        for sim in range(self.simulations):
            self.simulate()

        action = self.mcts.choose_action()

        return action

    def simulate(self) -> None:
        # selection
        leaf, path = self.mcts.select_leaf()
        pi, v = self.nn.predict(leaf.state)
        # expansion
        self.mcts.expand_leaf(leaf, pi)
        # backpropagation
        self.mcts.backpropagation(v, path)

