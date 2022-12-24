import collections.abc

def is_sequence(x):
    if isinstance(x, collections.abc.Sized):
        return True
    else:
        return False
