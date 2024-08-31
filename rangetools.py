import math
from functools import update_wrapper


def prod(r:range, /):
    '''Calculates the product of a range.'''
    if not r:
        return 1
    if 0 in r:
        return 0
    elif r.step == 1:
        value = math.factorial(r[-1])
        if (start := r.start) > 1:
            return value // math.factorial(start - 1)
        return value
    else:
        return math.prod(r)


def statistic(func, /):
    def function(r:range, /):
        if r:
            return func(r)
        else:
            raise ValueError(f"No {func.__name__} for empty range.")
    return update_wrapper(function, func, updated=('__annotations__',))


@statistic
def fmean(r) -> float:
    '''Calculates the mean of a range.'''
    return (r.start + r[-1]) / 2


def mean(r:range, /) -> float | int:
    return math.trunc(r) if (r := fmean(r)).is_integer() else r


@statistic
def median_high(r) -> int:
    '''Return the high median of a range.'''
    return r[len(r) // 2]


@statistic
def median_low(r) -> int:
    '''Return the low median of a range.'''
    i, mod = divmod(n := len(r), 2)
    return r[i] if mod else r[i - 1]


@statistic
def median(r) -> int | float:
    '''Return the median (middle value) of a range.'''
    i, mod = divmod(n := len(r), 2)
    return r[i] if mod else (r[i] + r[i - 1]) / 2


def sum(r) -> int:
    '''Calculates sum(range(*args)).'''
    return (r.start + r[-1]) * len(r) // 2 if r else 0


def setmethod(func, data={'r':range, 'x':range, 'return':bool}, /):
    func.__annotations__ |= data
    return func


@setmethod
def isdisjoint(r, x, /):
    '''Return True if two ranges have a null intersection.'''
    if r and x:
        if r.step <= -1:
            r = r[::-1]
        if x.step <= -1:
            x = x[::-1]
        return max(x.start, r.start) >= min(r[-1], x[-1])
    else:
        return True
        

@setmethod
def issubrange(r, x, /):
    '''Report whether another range contains this range.'''
    if not r:
        return True
    elif not x or (len(r) > 1 and r.step != x.step):
        return False
    else:
        return r.start in x and r[-1] in x


@setmethod
def issuperrange(r, x, /):
    '''Report whether this range contains another range.'''
    return issubrange(x, r)


@setmethod
def contains(r, x, /):
    '''Report wether elements of x range are in r range.'''
    if not x:
        return True
    elif not r or x.step % r.step:
        return False
    else:
        return x.start in r and x[-1] in r


@setmethod
def intersection(r, x, /):
    '''Returns the intersection of two ranges as a new range.'''
    start = max(r.start, x.start)
    stop = min(r.stop, x.stop)
    step = math.lcm(x.step, r.step)
    return range(start, stop, step)