from collections.abc import Callable, Generator
from operator import methodcaller, getitem
from itertools import accumulate, repeat
from array import array

class array(array):
    '''Array subclass allows to call the from unicode method using the lshift
    (<<) operator. Example:
    data = array('u')
    data << "Hello World!"
    print(data.tounicode())
    >> Hello World!
    '''
    __slots__ = ()
    __lshift__ = write = array.fromunicode


def char_increment(x:int = 1, /) -> Callable[(str,), str]:
    '''A callable that increments string chars x steps'''
    return methodcaller('translate', range(x, x + 1114112))


class LiteralAttr:
    __slots__ = ()
    @staticmethod
    def __getattr__(attr, /):
        return attr


def pyram(string, r:int, /, fillchar=' ', *, reverse:bool=False) -> Generator[str]:
    '''Generator of the lines of a pyram composed by the given string.
    string = the string that will make the pyram
    r = the number of rows of the pyram
    char = the char around the pyram, defulat an empty space

    Example:
    pyram

    prints:
    '\n'.join(pyram('*', 3))
      *   
     ***  
    ***** 

    if reverse is True, the pyram is returned in reverse order.
    '\n'.join(pyram('*', 3, reverse=True))
    ***** 
     ***  
      *   
    '''
    x = ((r * 2) - 1)
    n = len(string)
    
    if reverse:
        func, item = getitem, slice(-(n * 2))
        string *= x
    
    else:
        func, item = None, string * 2
    
    return map(methodcaller('center', x * n, fillchar),
        accumulate(repeat(item, r - 1), func, initial=string))


def square(string, n:int, /):
    '''Generator of the lines of a square composed by the given string'''
    return repeat(string * n, n)

del Callable, Generator