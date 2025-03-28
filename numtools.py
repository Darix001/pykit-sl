import operator as op, re

from numbers import Number
from bitarray import bitarray
from math import trunc, isqrt, ceil
from itertools import count, compress, islice
from collections.abc import Callable, Generator, Iterator


bitarray = bitarray('1')
operator_funcs = {
    '':op.add, '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, '**':op.pow,
    '//':op.floordiv,
    
    '&':op.and_, '|':op.or_, '^':op.xor, '%':op.mod, '@':op.matmul,
    
     '>':op.gt, '>=':op.ge, '<':op.lt, '<=':op.le, '==':op.eq, '!=':op.ne,
    }
re = re.compile(r'[-+]?(?:\d*\.*\d+)')


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


def gauss_sum(n:Number, /) -> Number:
    '''Sum of all numbers from start to stop.'''
    return n * (n + 1) // 2


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


def digits(x:int, /) -> int:
    '''Sum of all x's digits.'''
    while x:
        x, mod = divmod(x, 10)
        yield mod


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


def factors(n:int, /) -> Generator[int]:
    '''Returns all the factors of integer. This code is a variation of the
    source code from Stack Overflow: https://stackoverflow.com/a/6800214'''
    yield 1
    yield n

    for i in range(2, isqrt1(n)):
        div, mod = divmod(n, i)

        if not mod:
            yield i
            yield div

del Callable, Iterator, Generator