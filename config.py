"""
Configuration parameters
"""

# PLAY AND SELF-PLAY
EPISODES = 100
MCTS_SIMULATIONS = 500
MEMORY_SIZE = 100000
TURNS_BEFORE_TAU0 = 0
CPUCT = 1.41
EPSILON = 0.25
ALPHA = 0.05
TIMEOUT = 60
TAU = 5
TAU_ALPHA = 0.5
MAX_MOVES = 30

# NETWORK TRAINING AND HYPERPARAMETERS
BATCH_SIZE = 128
EPOCHS = 5
REG_CONST = 0.0001
LEARNING_RATE = 0.01
MOMENTUM = 0.9
TRAINING_LOOPS = 50
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
