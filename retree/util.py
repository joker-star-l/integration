import time
from loguru import logger
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} 执行耗时: {end - start:.6f} 秒")
        return result
    return wrapper
