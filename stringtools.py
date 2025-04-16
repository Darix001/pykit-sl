from .cachetools import Cache

from array import array
from collections import deque
from functools import wraps, partial
from dataclasses import dataclass, make_dataclass
from collections.abc import Callable, Iterator
from itertools import accumulate, repeat, filterfalse
from operator import methodcaller, getitem, itemgetter, attrgetter


STRITER = Iterator[str]
get_second = itemgetter(1)
attrgetters = Cache(attrgetter)


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


def pyram(string:str, r:int, /, fillchar:str=' ', increase:int=1, *,
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
        inc = (n := lenfd(string)) * increase
        func, item = getitem, slice(-inc)
        string *= (inc * (r - 1)) + n
    
    else:
        func, item = None, string * increase
    
    return accumulate(repeat(item, r - 1), func, initial=string)


def rreplace(string:str, oldsub:str, newsub:str, /, count=-1) -> str:
    if count:
        if count == 1:
            return replace_last(string, oldsub, newsub)
        else:
            return newsub.join(string.rsplit(oldsub, count))
    else:
        return string


def replace_last(string:str, oldsub:str, newsub:str, /) -> str:
    first, mid, last = string.rpartition(oldsub)
    return string if not mid else f"{first}{newsub}{last}"


def preffixer(string:str, /):
    return methodcaller('replace', '', string, 1)


def mal_formed_string():
    raise ValueError("Mal Formed String.")


Base = make_dataclass('Base', (('string', str),), slots=True, frozen=True)


@dataclass(frozen=True)
class Group(Base):
    __slots__ = ('indices', 'deepness')
    indices:range
    deepness:int

    def __repr__(self, /):
        span = self.span
        return (f"<Group object(match={self!s},"
            f"indices={self.span!r}, deepness={self.deepness!r})")

    def __str__(self, /):
        return self.string[self.start:self.stop]



def encloser(opening:str, closing:str=''):
    if not closing:
        opening, closing = opening

    return partial(Encloser, {opening:1, closing:-1}.get)


@dataclass(frozen=True)
class Encloser(Base):
    __slots__ = 'groups'
    string:str
    groups:list[Group]
    
    def __init__(self, func, string:str, /):
        it = enumerate(map(func, string))
        super().__init__(string)
        object.__setattr__(self, 'groups', groups := [])
        starts = []
        deep = 0

        for index, sign in filter(get_second, it):
            deep += sign
            if deep == -1:
                mal_formed_string()

            if sign == 1:
                starts.append(index)
            else:
                groups.append(
                    Group(string, range(starts.pop(), index + 1), deep)
                    )

        if deep:
           mal_formed_string()

        groups.sort(key=attrgetters['indices.start'])


    # def deep_map(func, /):
    #     diff = 0
    #     string = self.string

    #     for group in sorted(self.groups, key=attrgetters['step']):
    #         func(string[group.start:group.stop])

    
    def sortfunc(func, key=attrgetters['deep'], /):

        @wraps(func)
        def function(self, /):
            group = func(self.groups, key)
            return self.string[group.start:group.stop]

        return function

    @sortfunc
    def deepest(groups, key, /):
        return max(groups, key=key)

    @sortfunc
    def superficial(groups, key, /):
        return next(filterfalse(key, groups))


    def split(self, maxsplit:int=-1, /):
        data = []
        
        if maxsplit:
            string = self.string
            it = repeat(None) if maxsplit < 0 else repeat(None, maxsplit)
            indices = map(attrgetters['indices'],
             filterfalse(attrgetters['deepness'], self.groups)
             )
            start = 0

            for index, indices in zip(it, indices):
                data.append(string[:indices.start])
                string = string[indices.stop:]

            for x in it:
                data.append(string)
                break

        return data


    def deep_map():
        pass


del Callable, Iterator, STRITER, attrgetter, itemgetter