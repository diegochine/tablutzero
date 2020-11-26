"""
Configuration parameters
"""

# PLAY AND SELF-PLAY
EPISODES = 25
MCTS_SIMULATIONS = 400
MEMORY_SIZE = 100000
TURNS_BEFORE_TAU0 = 15
CPUCT = 1
EPSILON = 0.25
ALPHA = 0.2
TIMEOUT = 60
TAU = 1

# NETWORK TRAINING AND HYPERPARAMETERS
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.01
MOMENTUM = 0.9
TRAINING_LOOPS = 10
IN_SHAPE = (9, 9, 4)
OUT_SHAPE = (9, 9, 32)

HIDDEN_LAYERS = [
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
]

TOTAL_ITERATIONS = 1000

# OTHER
VERBOSE = 1
CURRENT_VERSION = 0
