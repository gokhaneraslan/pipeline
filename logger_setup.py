import logging
import sys

def setup_logger(log_level="INFO", log_file=None):
    """
    Configures and returns a logger instance.
    """
    logger = logging.getLogger("DataPipeline")
    logger.setLevel(log_level.upper())

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Stream Handler (for console output)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a') 
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
