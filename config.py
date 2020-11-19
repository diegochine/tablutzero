"""
Configuration parameters
"""

# PLAY AND SELF-PLAY
EPISODES = 10
MCTS_SIMULATIONS = 50
MEMORY_SIZE = 10000
TURNS_BEFORE_TAU0 = 10
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

# NETWORK TRAINING
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 10

HIDDEN_LAYERS = [
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3},
    {'filters': 128, 'kernel_size': 3}
]
