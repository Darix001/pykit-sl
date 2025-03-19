import operator as op, re

from numbers import Number
from typing import Optional
from types import MethodType
from bitarray import bitarray
from math import trunc, isqrt, ceil
from itertools import count, compress, islice
from collections.abc import Callable, Generator, Iterator


bitarray = bitarray('1')
operator_funcs = {
    '':op.add, '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv,
    '//':op.floordiv, '**':op.pow,
    
    '&':op.and_, '|':op.or_, '^':op.xor, '%':op.mod, '@':op.matmul,
    
     '>':op.gt, '>=':op.ge, '<':op.lt, '<=':op.le,
    '==':op.eq, '!=':op.ne
    }
re = re.compile(r'[-+]?(?:\d*\.*\d+)')


def partialop(symbol:str, value:Number, /) -> Callable:
    '''Partialize an operator function'''
    return MethodType(operator_funcs[symbol], value)


def rpartialop(symbol:str, value:Number):
    '''Partialize the second argument of an operator function'''
    func = operator_funcs[symbol]
    return lambda x, /: func(x, value)


def eval(string:str, /, dtype:Callable = int, start:int = 0) -> Number:
    '''Safely evaluates a numeric string expression.'''
    string = string.replace(' ', '')

    while stop := (string.find(')', start) + 1):
        start = string.rfind('(', 0, stop)
        substring = string[start:stop]
        numbers = map(dtype, re.findall(substring))
        x = next(numbers)
        operator = re.split(substring)

        del operator[0]

        operator = map(operator_funcs.get, operator)

        for n, operator in zip(numbers, operator):
            x = operator(x, n)

        string = string.replace(substring, f"{x!s}")

    return dtype(string)


def _cc2(compressor:Iterator, /) -> Iterator[int]:
    return compress(count(2), compressor)


def sieve(x:int, /) -> Iterator[int]:
    '''All Prime Numbers lower than x.'''    
    for x in _cc2(islice(data := bitarray * (x + 1), 2, isqrt1(x))):
        data[x*x::x] = 0
    
    del data[:2]
    return _cc2(data)


def gauss_sum(start:int, stop:int|None = None, /) -> int:
    '''Sum of all numbers from start to stop.'''
    s1 = start + 1
    if stop is None:
        return start * s1 // 2
    else:
        return trunc(((stop - s1) / 2) * (stop + start))


def collatz(x:Number, /) -> Generator[Number]:
    '''Yields all numbers of collatz formula until 1 is reached.'''
    while True:
        div, mod = divmod(x, 2)
        if div:
            yield (x := (((x * 3) + 1) if mod else div))
        else:
            break


def ndigits(x:int, /) -> int:
    '''Calculates len(str(x))'''
    i = trunc(0.30102999566398114 * (x.bit_length() - 1)) + 1
    return (10 ** i <= abs(x)) + i


def sumdigits(x:int, /, start:int = 0) -> int:
    '''Sum of all x's digits.'''
    while x:
        x, mod = divmod(x, 10)
        start += mod
    return start


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


def isqrt1(n:Number, /) -> int:
    '''Returns integer part of sqrt of number plus 1.'''
    return isqrt(n) + 1


def factors(n:int, /) -> set[int]:
    '''Returns all the factors of integer
    Extracted from Stack Overflow: https://stackoverflow.com/a/6800214'''
    x = set()

    for i in range(1, isqrt1(n)):
        div, mod = divmod(n, i)

        if not mod:
            x.add(i)
            x.add(div)
    return x


def primefactors(n:Number, /) -> Generator[Number]:
    while True:
        div, mod = divmod(n, 2)
        
        if mod:
            break
        
        yield 2
        n = div
         
    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3, isqrt1(n), 2):
         
        # while i divides n , print i ad divide n
        while True:
            div, mod = divmod(n, i)
            
            if mod:
                break
            
            yield i
            n = div
             
    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        yield n


del Callable, Iterator, Generator