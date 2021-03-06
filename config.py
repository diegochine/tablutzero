"""
Configuration parameters
"""

# PLAY AND SELF-PLAY
EPISODES = 50
MCTS_SIMULATIONS = 500
MEMORY_SIZE = 100000
TURNS_BEFORE_TAU0 = 5
CPUCT = 0.7
EPSILON = 0.2
ALPHA = 0.10
TIMEOUT = 60
TAU = 5
TAU_ALPHA = 0.5
MAX_MOVES = 10

# NETWORK TRAINING AND HYPERPARAMETERS
BATCH_SIZE = 64
EPOCHS = 10
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 50
IN_SHAPE = (9, 9, 4)
OUT_SHAPE = (9, 9, 32)

HIDDEN_LAYERS = [
    {'filters': 256, 'kernel_size': 3},
    {'filters': 256, 'kernel_size': 3},
    {'filters': 256, 'kernel_size': 3},
    {'filters': 256, 'kernel_size': 3},
    {'filters': 256, 'kernel_size': 3},
]

TOTAL_ITERATIONS = 1000

# OTHER
VERBOSE = 1
CURRENT_VERSION = 3
