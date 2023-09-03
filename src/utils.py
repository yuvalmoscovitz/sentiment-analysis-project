import numpy as np

def convert_numpy_int64(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_int64(value) for value in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj