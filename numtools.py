import operator, re, collections.abc as abc

from numbers import Number
from itertools import compress, islice, count
from math import isqrt, trunc, ceil
from bitarray import bitarray


bitarray = bitarray('1')
operators = {
    '':operator.add,
    '+':operator.add,
    '-':operator.sub,
    '*':operator.mul,
    '/':operator.truediv,
    '&':operator.and_,
    '|':operator.or_,
    '^':operator.xor,
    '%':operator.mod,
    '@':operator.matmul,
    '//':operator.floordiv,
    '**':operator.pow,
    '>':operator.gt,
    '>=':operator.ge,
    '<':operator.lt,
    '<=':operator.le,
    '==':operator.eq,
    '!=':operator.ne
    }
re = re.compile(r'[-+]?(?:\d*\.*\d+)')


def eval(string:str, /, dtype:abc.Callable = int, start:int = 0) -> Number:
    '''Safely evaluates a numeric string expression.'''
    string = string.replace(' ', '')

    while stop := (string.find(')', start) + 1):
        start = string.rfind('(', 0, stop)
        substring = string[start:stop]
        numbers = map(dtype, re.findall(substring))
        x = next(numbers)
        operator = re.split(substring)

        del operator[0]

        operator = map(operators.get, operator)

        for n, operator in zip(numbers, operator):
            x = operator(x, n)
        string = string.replace(substring, f"{x!s}")

    return dtype(string)


def sieve(x:int, /) -> abc.Iterator[int]:
    '''All Prime Numbers lower than x.'''
    data = bitarray * (x + 1)
    for x in compress(count(2), islice(data, 2, isqrt(x) + 1)):
        data[x*x::x] = 0
    del data[:2]
    return compress(count(2), data)


def gauss_sum(start:int, stop:int = None, /) -> int:
    '''Sum of all numbers from start to stop.'''
    if stop is None:
        return start * (start + 1) // 2
    return trunc(((stop - start + 1) / 2) * (stop + start))


def collatz(x:Number, /) -> abc.Generator[Number]:
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


def sumdigits(x:int, /, start:int = 0) -> Number:
    '''Sum of all x's digits.'''
    while x:
        x, mod = divmod(x, 10)
        start += mod
    return start


def nbytes(x:int, /) -> int:
    '''The amount of space in bytes that the integer would occupe.'''
    return ceil(x.bit_length() / 8)




del operator, abc, Number