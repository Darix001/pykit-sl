from array import array
from itertools import accumulate, repeat
from operator import methodcaller, getitem
from collections.abc import Callable, Iterator


STRITER = Iterator[str]


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


def pyram(string:str, r:int, /, fillchar:str=' ', increase=1, *,
    reverse:bool=False) -> STRITER:
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
    step = increase * 2
    x = (r * step) - (((increase - 1) * 2) + 1)
    n = len(string)
    
    if reverse:
        func, item = getitem, slice(-(n * step))
        string *= x
    
    else:
        func, item = None, string * step
    
    return map(methodcaller('center', x * n, fillchar),
        accumulate(repeat(item, r - 1), func, initial=string))


def square(string:str, n:int, /) -> STRITER:
    '''Generator of the lines of a square composed by the given string'''
    return repeat(string * n, n)


def stairs(string:str, r:int, /, fillchar:str=' ', increase=1, *,
    reverse:bool=False) -> STRITER:
    '''Generator of the lines of a stair composed by the given string.
    string = the string that will make the stairs
    r = the number of rows of the stairs
    char = the char around the stairs, defulat an empty space

    Example:
    stairs

    prints:
    '\n'.join(stairs('*', 3))
    *   
    **  
    *** 

    if reverse is True, the stairs is returned in reverse order.
    '\n'.join(stairs('*', 3, reverse=True))
    *** 
    **  
    *   
    '''
    
    if reverse:
        inc = (n := len(string)) * increase
        func, item = getitem, slice(-inc)
        string *= (inc * (r - 1)) + n
    
    else:
        func, item = None, string * increase
    
    return accumulate(repeat(item, r - 1), func, initial=string)


def rreplace(string:str, oldsub:str, newsub:str, /, count=-1) -> str:
    if count:
        if count == 1:
            first, mid, last = string.partition(oldsub)
            return string if not mid else f"{first}{newsub}{last}"
        else:
            return newsub.join(string.rsplit(oldsub, count))
    else:
        return string


def replacelast(string:str, oldsub:str, newsub:str, /) -> str:
    first, mid, last = string.rpartition(oldsub)
    return string if not mid else f"{first}{newsub}{last}"


del Callable, Iterator, STRITER