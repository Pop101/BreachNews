import inspect
import os
import tempfile
from typing import Any, Callable
import time
import pickle

CACHE_DIR = os.path.join(tempfile.gettempdir(), 'jpycache')

def clean_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # For each file in dir, delete it if it hasn't been accessed in the last 24 hours
    for file in os.listdir(CACHE_DIR):
        try:
            if os.stat(file).st_mtime < time.time() - (24 * 3600):
                os.remove(file)
        except FileNotFoundError:
            pass
    
def save(generator:Callable) -> Any:
    """
    Caches the value of the generation function in a temp file, using the src code as the key
    Generator must return an object that can be pickled
    """
    
    if not callable(filter):
        raise ValueError('Generator must be a callable function')
    if len(inspect.signature(generator).parameters) != 0:
        raise ValueError('Generator must take no arguments')
    
    clean_cache()
    
    src = inspect.getsource(generator)
    key = hash(src)
    path = os.path.join(CACHE_DIR, f"{key}.pkl")
    
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        value = generator()
        with open(path, 'wb') as f:
            pickle.dump(value, f)
        return value