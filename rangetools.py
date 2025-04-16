import math

from numbers import Number
from operator import attrgetter, neg, pos
from functools import update_wrapper as wrap, wraps, reduce

ROOT12 = math.sqrt(12)
rargs = attrgetter('start', 'stop', 'step')


def prod(r, /):
    '''Calculates the product of a range.'''
    if not r:
        return 1

    if 0 in r:
        return 0

    else:
        return math.prod(r)


def statistic(func, /):
    def function(r, /):
        if r:
            return func(r)
        else:
            raise ValueError(f"No {func.__name__} for empty range.")
    return wrap(function, func, updated=('__annotations__',))


def first_plus_last(r, /) -> Number:
    return r.start + r[-1]


@statistic
def fmean(r, /) -> Number:
    '''Calculates the mean of a range.'''
    return first_plus_last(r) / 2


def mean(r, /) -> Number | int:
    '''Calculates the mean of a range and returns integer when mean if is
    a whole numbers .'''
    return mean if (mean := fmean(r)) % 1 else math.trunc(mean)


@statistic
def median_high(r, /) -> int:
    '''Return the high median of a range.'''
    return r[len(r) // 2]


@statistic
def median_low(r, /) -> int:
    '''Return the low median of a range.'''
    i, mod = divmod(n := len(r), 2)
    return r[i] if mod else r[i - 1]

@statistic
def median(r, /) -> int | float:
    '''Return the median (middle value) of a range.'''
    i, mod = divmod(n := len(r), 2)
    return r[i] if mod else (r[i] + r[i - 1]) / 2


def rfsum(r, /) -> Number:
    '''Calculates sum(range(*args)).'''
    return first_plus_last(r) * len(r) / 2 if r else 0


def rsum(r, /) -> Number:
    '''Calculates sum(range(*args)).'''
    return first_plus_last(r) * len(r) // 2 if r else 0


def minmax(func, /):
    @statistic
    def function(r, /):
        return func(r, r.step > 0)
    return function

@minmax
def rmin(r, growing, /) -> Number:
    return r.start if growing else r[-1]

@minmax
def rmax(r, growing, /) -> Number:
    return r.start if growing else r[-1]

@minmax
def argmin(r, growing, /) -> Number:
    return 0 if growing else len(r) - 1

@minmax
def argmax(r, growing, /) -> Number:
    return len(r) - 1 if growing else 0



def isdisjoint(r, x, /) -> bool:
    '''Return True if two ranges have a null intersection.'''
    if r and x:
        if r.step <= -1:
            r = r[::-1]
        if x.step <= -1:
            x = x[::-1]
        return max(x.start, r.start) >= min(r[-1], x[-1])
    else:
        return True


def issubrange(r, x, /) -> bool:
    '''Report whether another range contains this range.'''
    if not r:
        return True
    elif not x or (len(r) > 1 and r.step != x.step):
        return False
    else:
        return r.start in x and r[-1] in x


def issuperrange(r, x, /) -> bool:
    '''Report whether this range contains another range.'''
    return issubrange(x, r)


def subcontains(r, x, /) -> bool:
    '''Report wether elements of x range are in r range.'''
    if not x:
        return True
    elif not r or x.step % r.step:
        return False
    else:
        return x.start in r and x[-1] in r


def and_(r, x, /):
    '''The algorithm to get the first intersection point was provided
    by chatGPT
    #By assigning r.start as the final start variable we save few lines
    of code.
    start = r.start 
    b = x.start
    s1 = r.step
    s2 = x.step
    diff = Difference between starts
    g = greatest common divissor between steps'''
    
    cls = type(r)
    start, b = r.start, x.start

    if (ss := (step := r.step) == (s2 := x.step)) == 1:
        if start < b:
            start = b
    
    else:
        s1 = step
    
        if ss:
            g = step
        else:
            if s1 < (g := s2):
                step, g = g, step

            if step % g:
                step *= g
                g = 1

        # If g doesn't divide diff, no solution
        if (diff := b - start) % g:
            return cls(zero := start * 0, zero)

        # Solve for the smallest non-negative n
        # Modular inverse of (s1 // g) mod (s2 // g)
        mod_inv = pow(s1 // g, -1, s2g := s2 // g)
        
        # First intersection point
        start += ((diff // g * mod_inv) % s2g) * s1

    if (stop := x.stop) > (rstop := r.stop):
        stop = rstop

    return cls(start, stop, step)


def intersection(*args):
    if not args:
        return range(0, 0)

    cls = type(self := args[0])
    if all(args):
        starts, stops, steps = zip(*map(rargs, args))
        
        if steps.count(1) == len(steps):
            return cls(min(starts), max(stops))

        else:
            return reduce(and_, args)

    return cls(zero := self.start * 0, zero)


def variance_func(func, /):
    def function(x, /):
        if self:
            n = len(self)
            return func(self.step, ((n * n) - 1) / n)
        else:
            return math.nan
    return function


@variance_func
def std(d, n, /):
    return (d / ROOT12) * math.sqrt(n)

@variance_func
def var(d, n, /):
    return ((d * d) / 12) * n


def opfunc(func, /):
    def func(r, /): type(r)(*map(func, rargs(r)))
    func.__doc__ = f'Returns {(name := func.__name__)} version of range r'
    func.__name__ = name
    return func


def invert(r, /):
    return type(r)(~r.start, ~r.stop, -r.step)


def advanced(r, n:Number=1, /):
    '''Returns a new range that walks n steps to the right (default n=1).
    If n is negative, advance left.
    advance(range(10), 1) is, in the context of a sequence, roughly equivalent
    to:
    numpy.arange(10) + 1 and [x + 1 for x in range(10)]'''
    if r:
        n *= (step := r.step)
        return cls(r.start + steps, stop + steps, step)
    else:
        return r


def expanded(r, n:Number=1, /):
    '''Returns a new range that expanded n steps to the right (default n=1).
    If n is negative, expand left.

    This method does not care if the range is empty,
    so expand(range(0)) returns range(1).

    However, it can return empty range if there is a big enough different
    between the start and the stop values relative to the step.
    
    For instance:
    expand(range(30, 10)) returns range(30, 11) due to the positive step.
    With this in count, please use this method with care.

    '''
    steps = n * (step := r.step)
    start, stop = r.start, r.stop
    if n > 0:
        stop += steps
    else:
        start += steps
    return type(r)(start, stop, step)


__and__ = and_

neg = __neg__ = opfunc(neg)

pos = __pos__ = opfunc(pos)

__invert__ = invert