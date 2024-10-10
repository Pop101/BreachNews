from typing import Any
from types import GeneratorType

class hashdict(dict):
    """
    hashable dict implementation, suitable for use as a key into
    other dicts.

        >>> h1 = hashdict({"apples": 1, "bananas":2})
        >>> h2 = hashdict({"bananas": 3, "mangoes": 5})
        >>> h1+h2
        hashdict(apples=1, bananas=3, mangoes=5)
        >>> d1 = {}
        >>> d1[h1] = "salad"
        >>> d1[h1]
        'salad'
        >>> d1[h2]
        Traceback (most recent call last):
        ...
        KeyError: hashdict(bananas=3, mangoes=5)

    based on answers from
       http://stackoverflow.com/questions/1151658/python-hashable-dicts

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__post_init__()
    
    def __post_init__(self):
        self.__key = tuple(sorted(self.items()))

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(f'{str(i[0])}={repr(i[1])}' for i in self.__key)})"

    def __hash__(self):
        return hash(self.__key)
    
    # def __setitem__(self, key, value):
    #     raise TypeError(f"{self.__class__.__name__} does not support item assignment")
    
    def __delitem__(self, key):
        raise TypeError(f"{self.__class__.__name__} does not support item assignment")
    
    def clear(self):
        raise TypeError(f"{self.__class__.__name__} does not support item assignment")
    
    def pop(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} does not support item assignment")
    
    def popitem(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} does not support item assignment")
    
    def setdefault(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} does not support item assignment")
    
    def update(self, *args, **kwargs):
        raise TypeError(f"{self.__class__.__name__} does not support item assignment")
    
    def __add__(self, right):
        # Creates a new object, thus allowing mutation
        result = hashdict(self)
        dict.update(result, right)
        return result
    
    def __lt__(self, other):
        """This function should not be used, yet is inexplicably called by pandas"""
        if not isinstance(other, (hashdict, dict)):
            return TypeError(f"Cannot compare hashdict to {type(other)}")
        if isinstance(other, dict):
            other = hashdict(other)
        return hash(self.__key) < hash(other.__key)
    
    def __eq__(self, other):
        if not isinstance(other, (hashdict, dict)):
            return False
        if isinstance(other, dict):
            other = hashdict(other)
        return hash(self.__key) == hash(other.__key)

    def __reduce__(self):
        return (self.__class__, (dict(self),))
    
    def __repr__(self) -> str:
        # Overwrite repr to allow for `eval(repr(hashdict))`
        return f"{self.__class__.__name__}({str(dict(self))})"

    def __str__(self) -> str:
        # Overwrite str to be human-readable dict format
        return str(dict(self))
    
def is_hashable(obj:Any)->bool:
    try:
        hash(obj)
        return True
    except TypeError:
        return False

def make_hashable(obj:Any) -> Any:
    """
    Recursively attempts to convert a nested object into a hashable object
    Dicts are converted to hashdicts, lists to tuples, and sets to frozensets
    Arbitrary objects are checked for hashability
    """
    

    if isinstance(obj, dict):
        return hashdict((k, make_hashable(v)) for k, v in obj.items())
    elif isinstance(obj, list):
        return tuple(make_hashable(v) for v in obj)
    elif isinstance(obj, set):
        return frozenset(make_hashable(v) for v in obj)
    elif isinstance(obj, GeneratorType):
        return tuple(make_hashable(v) for v in obj)
    elif is_hashable(obj):
        return obj
    else:
        raise ValueError(f"Object {obj} is not hashable")