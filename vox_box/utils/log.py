from functools import wraps
import logging
import time

logger = logging.getLogger(__name__)


def log_method(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__} method in {self.__class__.__name__}")
        result = func(self, *args, **kwargs)
        delay = time.time() - start_time
        logger.info(
            f"Finished {func.__name__} method in {self.__class__.__name__}, delay: {delay:.2f}s"
        )
        return result

    return wrapper
