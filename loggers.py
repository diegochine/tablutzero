import logging


def setup_logger(name, file, level=logging.INFO):
    formatter = logging.Formatter('{asctime} {levelname} {message}', style='{')
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger


logger_mcts = setup_logger('MCTS', 'logs/MCTS.log')
logger_train = setup_logger('train', 'logs/main.log')
logger_nnet = setup_logger('neuralnet', 'logs/nnet.log')
logger_memory = setup_logger('memory', 'logs/memory.log')
