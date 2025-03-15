import math, itertools as it
from operator import attrgetter
from functools import update_wrapper as wrap, wraps, reduce

ROOT12 = math.sqrt(12)
rargs = attrgetter('start', 'stop', 'step')


def slicer(cls, /):
    '''Decorator for range-like classes'''
    cls.__init__ = wrap(lambda obj, /, *args: func(obj, slice(*args)),
        cls.__init__)
    return cls


def prod(r:range, /):
    '''Calculates the product of a range.'''
    if not r:
        return 1

    if 0 in r:
        return 0

    else:
        return math.prod(r)


def statistic(func, /):
    def function(r:range, /):
        if r:
            return func(r)
        else:
            raise ValueError(f"No {func.__name__} for empty range.")
    return wrap(function, func, updated=('__annotations__',))

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


def rfsum(r) -> int:
    '''Calculates sum(range(*args)).'''
    return (r.start + r[-1]) * len(r) / 2 if r else 0

def rsum(r) -> int:
    '''Calculates sum(range(*args)).'''
    return (r.start + r[-1]) * len(r) // 2 if r else 0


def minmax(func, /):
    @statistic
    def function(r, /):
        return func(r, r.step > 0)
    return function

@minmax
def rmin(r, growing, /) -> int:
    return r.start if growing else r[-1]

@minmax
def rmax(r, growing, /) -> int:
    return r.start if growing else r[-1]

@minmax
def argmin(r, growing, /) -> int:
    return 0 if growing else len(r) - 1

@minmax
def argmax(r, growing, /) -> int:
    return len(r) - 1 if growing else 0


def rangesetfunc(func, data={'r':range, 'x':range, 'return':bool}, /):
    func.__annotations__ |= data
    return func

@rangesetfunc
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

@rangesetfunc
def issubrange(r, x, /):
    '''Report whether another range contains this range.'''
    if not r:
        return True
    elif not x or (len(r) > 1 and r.step != x.step):
        return False
    else:
        return r.start in x and r[-1] in x

@rangesetfunc
def issuperrange(r, x, /):
    '''Report whether this range contains another range.'''
    return issubrange(x, r)

@rangesetfunc
def contains(r, x, /):
    '''Report wether elements of x range are in r range.'''
    if not x:
        return True
    elif not r or x.step % r.step:
        return False
    else:
        return x.start in r and x[-1] in r

@rangesetfunc
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
        mod_inv = (1 / (s1 // g)) % (s2g := s2 // g)
        
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
    def function(x:range, /):
        if self:
            return func(self.step, (((n := len(self)) **2) - 1) / n)
        else:
            return math.nan
    return function


@variance_func
def std(d, n, /):
    return (d / ROOT12) * math.sqrt(n)

@variance_func
def var(d, n, /):
    return (d**2 / 12) * n