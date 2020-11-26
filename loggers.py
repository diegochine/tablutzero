import logging

disabled = {'player': False,
            'mcts': False,
            'train': False,
            'nnet': False,
            'mem': False}


def setup_logger(name, file, level=logging.INFO):
    formatter = logging.Formatter('{asctime} {levelname} {message}', style='{')
    file_handler = logging.FileHandler(file, mode='w')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


logger_mcts = setup_logger('MCTS', 'logs/MCTS.log')
logger_mcts.disabled = disabled['mcts']
logger_train = setup_logger('train', 'logs/main.log')
logger_train.disabled = disabled['train']
logger_nnet = setup_logger('neuralnet', 'logs/nnet.log')
logger_nnet.disabled = disabled['nnet']
logger_memory = setup_logger('memory', 'logs/memory.log')
logger_memory.disabled = disabled['mem']
logger_player = setup_logger('player', 'logs/player.log')
logger_player.disabled = disabled['player']
