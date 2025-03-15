import operator as op, re, collections.abc as abc, itertools as it, math
from types import MethodType
from numbers import Number
from bitarray import bitarray


bitarray = bitarray('1')
operator_funcs = {
    '':op.add, '+':op.add, '-':op.sub, '*':op.mul, '/':op.truediv,
    '//':op.floordiv, '**':op.pow,
    
    '&':op.and_, '|':op.or_, '^':op.xor, '%':op.mod, '@':op.matmul,
    
     '>':op.gt, '>=':op.ge, '<':op.lt, '<=':op.le,
    '==':op.eq, '!=':op.ne
    }
re = re.compile(r'[-+]?(?:\d*\.*\d+)')


def partialop(symbol:str, value:Number, /):
    '''Partialize an operator function'''
    return MethodType(operator_funcs[symbol], value)


def rpartialop(symbol:str, value:Number):
    '''Partialize the second argument of an operator function'''
    func = operator_funcs[symbol]
    return lambda x, /: func(x, value)


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

        operator = map(operator_funcs.get, operator)

        try:
            for n, operator in zip(numbers, operator):
                x = operator(x, n)
        except Exception as e:
            
        finally:
            pass

        string = string.replace(substring, f"{x!s}")

    return dtype(string)


def sieve(x:int, /) -> abc.Iterator[int]:
    '''All Prime Numbers lower than x.'''
    data = bitarray * (x + 1)
    for x in it.compress(it.count(2), it.islice(data, 2, math.isqrt(x) + 1)):
        data[x*x::x] = 0
    del data[:2]
    return it.compress(it.count(2), data)


def gauss_sum(start:int, stop:int = None, /) -> int:
    '''Sum of all numbers from start to stop.'''
    if stop is None:
        return start * (start + 1) // 2
    return math.trunc(((stop - start + 1) / 2) * (stop + start))


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
    i = math.trunc(0.30102999566398114 * (x.bit_length() - 1)) + 1
    return (10 ** i <= abs(x)) + i


def sumdigits(x:int, /, start:int = 0) -> Number:
    '''Sum of all x's digits.'''
    while x:
        x, mod = divmod(x, 10)
        start += mod
    return start


def nbytes(x:int, /) -> int:
    '''The amount of space in bytes that the integer would occupe.'''
    return math.ceil(x.bit_length() / 8)


def lcm2(x:Number, y:Number, /) -> Number:
    '''Returns the lcm of passed numbers. Unlike like math.lcm, it works with
    any kind of numbers'''
    if x != y:
        if x < y:
            x, y = y, x

        if x % y:
            return x * y

    return x
