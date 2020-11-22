"""
Configuration parameters
"""

# PLAY AND SELF-PLAY
EPISODES = 10
MCTS_SIMULATIONS = 50
MEMORY_SIZE = 10000
TURNS_BEFORE_TAU0 = 15
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8
TIMEOUT = 60

# NETWORK TRAINING AND HYPERPARAMETERS
BATCH_SIZE = 256
EPOCHS = 5
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10
IN_SHAPE = (9, 9, 4)
OUT_SHAPE = (9, 9, 32)

HIDDEN_LAYERS = [
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
]

TOTAL_ITERATIONS = 5

# OTHER
VERBOSE = 1
