try:
    import builtins
except ImportError:
    # Python 2
    import __builtin__ as builtins

def print(*args, **kwargs):
    return builtins.print("[cs231n-mammogram]", *args, **kwargs)


