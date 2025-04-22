from .methodtools import setname_factory, builtin_magic, name_wrap, unassigned

from array import array
from functools import wraps, partial
from itertools import accumulate, repeat, islice
from dataclasses import dataclass
from operator import methodcaller, getitem, itemgetter
from collections.abc import Callable, Iterator, Generator


STRITER = Iterator[str]
item1 = itemgetter(1)


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
        inc = (n := len(string)) * increase
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


@dataclass(frozen=True)
class Sub:
    __slots__ = 'string', 'indices'
    string:str
    indices:range
    maketrans = str.maketrans
    
    def __str__(self, /):
        indices = self.indices
        return self.string[indices.start:indices.stop]

    def __getitem__(self, key, /):
        string = self.string
        if type(key := self.indices[key]) is int:
            return string[key]
        else:
            type(self)(string, key)

    @builtin_magic
    def __len__(func, /):
        return lambda self, /: func(self.indices)

    __bool__ = __len__

    def __iter__(self, /):
        it = iter(string := self.string)
        indices = self.indices
        
        stop = indices.stop
        
        if start := indices.start:
            it.__setstate__(start)
            stop -= start

        return it if stop > len(string) else islice(it, stop)


    def __reversed__(self, /):
        return map(getitem, repeat(self.string), self.indices)

    def __contains__(self, string, /):
        i = self.indices
        return self.string.find(string, i.start, i.stop) != -1

    def __eq__(self, s, /):
        i = self.indices
        if isinstance(s, type(string := self.string)):
            return len(s) == len(i) and string.startswith(s, i.start)
        
        elif isinstance(s, Sub):
            return s.string is string and i == s.indices
        
        else:
            return NotImplemented


    def stringfunc(name, /):
        
        def func(self, sub:str, start:int=0, stop:int|None=None, /):
            indices = self.indices[start:stop]
            method = getattr(self.string, name)
            return method(sub, indices.start, indices.stop)

        func.__doc__ = getattr(str, name).__doc__
        return func

    endswith = startswith = count = setname_factory(stringfunc)

    
    @setname_factory
    def index(name, factory=stringfunc, /):
        
        @wraps(func := factory(name))
        def function(*args):
            return args[0].indices.index(func(*args))

        return function

    rindex = index
    

    @setname_factory
    def find(name, factory=stringfunc, /):
        
        @wraps(fn := factory(name))
        def function(*args):
            if (i := fn(*args)) != -1:
                return args[0].indices.index(i)
            else:
                return i

        return function

    rfind = find

    
    def finditer(self, sub:str, start:int=0, stop:int|None=None, /
        ) -> Generator[int]:
        
        i = self.indices[start:stop]
        start, stop = i.start, i.stop
        sub_size = len(sub)
        find = self.string.find
        
        while (start := find(sub, start, stop)) != -1:
            yield start
            start += sub_size

    
    
    
    def splitter(gen:Generator, /):
        
        @name_wrap(gen)
        def function(self, sub:str, maxsplit:int=-1, /):
            if maxsplit:
                string = self.string
                g = gen(self.indices, string.find, sub, maxsplit)
                return [*map(type(self), repeat(string), map(range, g, g))]
            
            else:
                return [self]
        
        return function
            

    @splitter
    def split(indices, finder, sub, maxsplit, /):
        start = indices.start
        stop = indices.stop
        sub_size = len(sub)
        it = repeat(None) if maxsplit < 0 else repeat(None, maxsplit)

        for _ in it:
            yield start
    
            if (start := finder(sub, start, stop)) == -1:
                break
            else:
                yield start
                start += sub_size

        yield stop
            
    # @splitfunc
    # def rsplit(start, stop, string, sep_size, sep, it, /):
    #     indices = []
    #     for _ in it:
    #         yield
    #         if (value := string.rfind(sep, start, stop)) != -1:
    #             indices.append(value)
    #             indices.append(stop)
    #             stop = value - sep_size
    #         else:
    #             break

    #     return reversed(indices)




    def removeprefix(self, prefix, /):
        '''Return a Sub with the given prefix string removed if present.

        If the string starts with the prefix string, return string[len(prefix):].
        Otherwise, return a copy of the original string.'''
        
        string = self.string
        indices = self.indices
        if preffix and string.startswith(prefix, indices.start, indices.stop):
            return type(self)(string, i[len(prefix):])

        return self


    def removesuffix(self, suffix, /):
        '''Return a Sub with the given suffix string removed if present.

        If the string ends with the suffix string,
        return string[:-len(suffix)].

        Otherwise, return a copy of the original string.'''
        
        string = self.string
        indices = self.indices
        if suffix and string.endswith(suffix, indices.start, indices.stop):
            return type(self)(string, i[:-len(suffix)])

        return self


    def partitionfunc(name, /) -> tuple:

        @unassigned
        def function(self, sep, /):
            indices = self.indices
            string = self.string
            start, stop = indices.start, indices.stop

            if value := getattr(string, name)(sep, start, stop):
                cls = type(self)
                return (cls(string, range(start, value)), sep,
                    cls(string, range(value + len(sep), stop)))
            else:
                return (self, '', '')

        return function


    partition, rpartition = map(partitionfunc, ('find', 'rfind'))


    @classmethod
    def new(cls, string, start:int=0, stop:int=-1, /):
        return cls(string, range(start, len(string) if stop < 0 else stop))

    @setname_factory
    def isupper(name, /):
        func = methodcaller(name)
        return lambda self, /: all(map(func, self))

    isascii = islower = isprintable = istitle = isspace = isdecimal = isupper
    isdigit = isnumeric = isalpha = isalnum = isidentifier = isupper

    
    def striper(func, /):
        
        @name_wrap(func)
        def function(self, chars:str=None, /):
            if chars is None:
                chars = whitespace
            
            elif not chars:
                return self

            else:
                i = self.indices
                return func(self.string, i.start, i.stop, chars)

        return function


    @striper
    def strip(string, start, stop, chars, /):
        pass


    @striper
    def lstrip(string, start, stop, chars, /):
        pass


    @striper
    def rstrip(string, start, stop, chars, /):
        pass


def nested_group_error(strip:bool, p1:str, p2:str, index:int, /) -> ValueError:
    return ValueError(f'{p1}{p2}, at column {index - strip!r}')
    
        
def nested_grouper(chars:str, /):
    op, cl = chars

    def function(string:str, /, strip=False):
        groups = []
        it = filter(item1, enumerate(map({op:1, cl:-1}.get, string), strip))
        it2 = map(len, repeat(starts := [])) #Iterator for DRY code
        stop_add = -1 if strip else 1
        error = partial(nested_group_error, strip)

        #depth = number of openings == len(starts)
        for (index, sign), depth in zip(it, it2):

            #An opening
            if sign == 1:
                if depth >= len(groups):
                    groups.append([])

                starts.append(index)

            #If there is a closing, there must be an opening catched before
            elif depth:
                depth -= 1
                groups[depth].append(string[starts.pop():index + stop_add])

            #At this point, the string has more closings than openings.
            else:
                raise error("unmatched ", cl, index)


        if depth:
           raise error(op, " was never closed", starts.pop())
        else:
            return groups

    return function


del Callable, Iterator, STRITER, itemgetter, Generator