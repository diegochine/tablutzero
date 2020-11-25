"""
Configuration parameters
"""

# PLAY AND SELF-PLAY
EPISODES = 100
MCTS_SIMULATIONS = 500
MEMORY_SIZE = 50000
MIN_MEMORIES = 20000
TURNS_BEFORE_TAU0 = 15
CPUCT = 1.41
EPSILON = 0.25
ALPHA = 0.2
TIMEOUT = 60
TAU = 1

# NETWORK TRAINING AND HYPERPARAMETERS
BATCH_SIZE = 512
EPOCHS = 10
REG_CONST = 0.0001
LEARNING_RATE = 0.05
MOMENTUM = 0.9
TRAINING_LOOPS = 200
IN_SHAPE = (9, 9, 4)
OUT_SHAPE = (9, 9, 32)

HIDDEN_LAYERS = [
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
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
