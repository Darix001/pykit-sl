import operator as op, collections.abc as abc, itertools as it, re, math

from numbers import Number
from functools import cache
from math import isqrt, ceil
from bitarray import bitarray
from itertools import count, compress, islice

bitarray = bitarray('1')

op_funcs = {
    '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, '**':op.pow,
    '//':op.floordiv, '&':op.and_, '|':op.or_, '^':op.xor, '%':op.mod,
     '>':op.gt, '>=':op.ge, '<':op.lt, '<=':op.le, '==':op.eq, '!=':op.ne,
    }


math_funcs = (
    'asin', 'sin', 'sinh', 'asinh', #sin functions
    'acos', 'cosh', 'acosh', 'cos', #cos functions
    'log', 'log10', 'log2', 'log1p', #log functions
    'tan', 'tanh', 'atan', 'atanh', 'atan2', #tan funcs
    'trunc', 'ceil', 'floor', 'fabs', 'sqrt', 'isqrt' #other funcs
    )

math_funcs = dict(zip(math_funcs, op.attrgetter(*math_funcs)(math)),
    abs=abs, round=round, pow=pow)


def simple_eval(string:str, /, dtype=int,
    re=re.compile(r'[-+]?(?:\d*\.*\d+)')) -> Number:
    start = string
    numbers = map(dtype, re.findall(substring))
    x = next(numbers)
    operator = re.split(substring)
    del operator[0]
    funcs = map(op_funcs.get, operator)
    
    for func, x in zip(funcs, numbers):
        x = func(x, n)

    return x



def aeval(string:str, /, dtype:abc.Callable=int, start:int=0) -> Number:
    '''Safely evaluates a numeric string expression.'''
    while (stop := string.find(')', start)) != -1:
        start = string.rfind('(', 0, stop)
        if sstart := start + 1:
            result = simple_eval(string[start:stop], dtype)
            if string[start - 1].isalpha():
                pass
            else:
                string = f"{string[:start]}{result}{string[stop:]}"
        else:
            raise ValueError("MalFormed String")
            

def cc2(compressor:abc.Iterator, /) -> abc.Iterator[int]:
    '''shortcut for: itertools.compress(itertools.count(2), compressor)'''
    return it.compress(it.count(2), compressor)


def sieve(x:int, /) -> abc.Iterator[int]:
    '''All Prime Numbers lower than x.'''
    data = bitarray * (x + 1)
    for x in _cc2(it.islice(data, 2, sqrt1(x) + 1)):
        data[x*x::x] = 0
    
    del data[:2]
    return _cc2(data)


def gauss_sum(n:Number, /) -> Number:
    '''Sum of all numbers from start to stop.'''
    return (n * (n + 1)) // 2


def collatz(x:Number, /) -> abc.Generator[Number]:
    '''Yields all numbers of collatz formula until 1 is reached.'''
    while True:
        div, mod = divmod(x, 2)
        if div:
            yield (x := (((x * 3) + 1) if mod else div))
        else:
            break

        
def nbytes(x:int, /) -> int:
    '''The amount of space in bytes that the integer would occupe.'''
    return ceil(x.bit_length() / 8)


def lcm2(x:Number, y:Number, /) -> Number:
    '''Returns the lcm of passed numbers. Unlike like lcm, it works with
    any kind of numbers'''
    if x != y:
        if x < y:
            x, y = y, x

        if x % y:
            return x * y

    return x

del math, Number