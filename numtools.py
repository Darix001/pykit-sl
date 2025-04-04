import operator as op, collections.abc as abc, itertools as it

from numbers import Number
from bitarray import bitarray
from re import compile as recompile
from math import log10, isqrt, ceil, trunc
from itertools import count, compress, islice

DIGITS = range(10)

bitarray = bitarray('1')
operator_funcs = {
    '':op.add, '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv, '**':op.pow,
    '//':op.floordiv,
    
    '&':op.and_, '|':op.or_, '^':op.xor, '%':op.mod, '@':op.matmul,
    
     '>':op.gt, '>=':op.ge, '<':op.lt, '<=':op.le, '==':op.eq, '!=':op.ne,
    }
RE = recompile(r'[-+]?(?:\d*\.*\d+)')


def eval(string:str, /, dtype:abc.Callable=int, start:int=0) -> Number:
    '''Safely evaluates a numeric string expression.'''
    string = string.replace(' ', '')

    while stop := string.find(')', start) + 1:
        start = string.rfind('(', 0, stop)
        substring = string[start:stop]
        numbers = map(dtype, RE.findall(substring))
        x = next(numbers)
        operator = RE.split(substring)

        del operator[0]

        op_funcs = map(operator_funcs.get, operator)

        for op_func, x in zip(op_funcs, numbers):
            x = op_func(x, n)

        string = string.replace(substring, f"{x!s}")

    return dtype(string)


def cc2(compressor:abc.Iterator, /) -> abc.Iterator[int]:
    '''shortcut for: itertools.compress(itertools.count(2), compressor)'''
    return it.compress(it.count(2), compressor)


def sieve(x:int, /) -> abc.Iterator[int]:
    '''All Prime Numbers lower than x.'''
    for x in _cc2(it.islice(data := bitarray * (x + 1), 2, isqrt1(x))):
        data[x*x::x] = 0
    
    del data[:2]
    return _cc2(data)


def gauss_sum(n:Number, /) -> Number:
    '''Sum of all numbers from start to stop.'''
    return n * (n + 1) // 2


def collatz(x:Number, /) -> abc.Generator[Number]:
    '''Yields all numbers of collatz formula until 1 is reached.'''
    while True:
        div, mod = divmod(x, 2)
        if div:
            yield (x := (((x * 3) + 1) if mod else div))
        else:
            break


class Digits(abc.Sequence):
    '''Emulates a list of integers composed by the digits of a number.
    This class is intended for dealing with very large integers numbers.'''
    __slots__ = ('_x', '_hash')

    def __new__(cls, x:int, /):
        if x in DIGITS:
            return (x,)
        else:
            self = super().__new__(cls)
            self._x = abs(x)
            return self
    
    def __repr__(self, /):
        return f"{type(self).__name__}({self._x!r})"

    def __len__(self, /):
        return trunc(log10(self._x)) + 1

    def __bool__(self, /):
        return True

    def __reversed__(self, /):
        x = self._x
        while x:
            x, mod = divmod(x, 10)
            yield mod

    def __getitem__(self, index, /):
        if index >= 0:
            index -= len(self)

        if div := self._x // (10 ** ~index):
            return div % 10
        else:
            raise IndexError("Digit object Index out of range.")


    def __contains__(self, digit, /):
        return digit in DIGITS and digit in reversed(self)

    def __hash__(self, /):
        if (hash_value := getattr(self, '_hash', None)) is not None:
            self._hash = hash_value = hash((self._x,))
        return hash_value

    def __mul__(self, times, /):
        if times > 0:
            base = 10 ** len(self)
            original = x = self._x
            
            for _ in range(times - 1):
                x =  x * base + original
            
            return type(self)(x)
        else:
            return ()


    def __add__(self, obj, /):
        if (cls := type(self)) is type(obj):
            return cls((self._x * (10 ** len(obj))) + obj._x)
        else:
            return NotImplemented

    def index(self, digit:int, /, start:int=0, stop:int|None=None) -> int:
        return super().index(DIGITS.index(digit), start, stop)

    def count(self, digit:int, /) -> int:
        return super().count(digit) if digit in DIGITS else 0

    @classmethod
    def fromiterable(cls, iterable:abc.Iterable[int], /):
        #Check if the sequence has items
        for start in (iterable := iter(iterable)):
            break
        else:#return empty tuple if iterable has no items.
            return ()
        
        for number in iterable:
            mul = 10 if number in DIGITS else 10 ** (trunc(log10(number)) + 1)
            start = start * mul + number
                
        else:
            return (start,)

        return cls(start)
            


class DigitList(Digits, abc.MutableSequence):
    __slots__ = ()

    def __iadd__(self, obj, /):
        if isinstance(obj, Digits):
            self._x = (self._x * (10 ** len(obj))) + obj._x
            return self
        else:
            return NotImplemented

    def append(self, digit:int):
        if digit in DIGITS:
            self._x = self._x * 10 + digit
        else:
            raise ValueError("DigitList item must be in range(10)")

    def reverse(self, /):
        first = next(digits := reversed(self))
        for digit in digits:
            first = first * 10 + digit
        self._x = first


    def pop(self, /) -> Number:
        self._x, digit = divmod(self._x, 10)
        return digit

    def extend(self, data:abc.Iterable[int], /):
        x = self._x
        if isinstance(data, Digits):
            x = x * (10 ** len(data)) + data._x        
        else:
            for number in iterable:
                mul = 10
                if number not in DIGITS:
                    mul **= (trunc(log10(number)) + 1)
                x = x * mul + number
        self._x = x

        
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


def factors(n:int, /) -> abc.Generator[int]:
    '''Returns all the factors of integer. This code is a variation of the
    source code from Stack Overflow: https://stackoverflow.com/a/6800214'''
    yield 1
    yield n

    for i in range(2, isqrt1(n)):
        div, mod = divmod(n, i)

        if not mod:
            yield i
            yield div

print(Digits(123456))