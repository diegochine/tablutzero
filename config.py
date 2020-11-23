"""
Configuration parameters
"""

# PLAY AND SELF-PLAY
EPISODES = 100
MCTS_SIMULATIONS = 500
MEMORY_SIZE = 50000
TURNS_BEFORE_TAU0 = 15
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.2
TIMEOUT = 60
TAU = 2

# NETWORK TRAINING AND HYPERPARAMETERS
BATCH_SIZE = 256
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
