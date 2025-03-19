from timeit import Timer
from numbers import Integral
from functools import partial
from collections import Counter
from collections.abc import Callable
from more_itertools import repeatfunc


fastest_timer = partial(min, key=Timer.timeit)


def with_args(args, /, *funcs, iterations:Integral=100) -> dict[Callable, int]:
	funcs = (partial(func, *args) for func in funcs)
	return benchsort(*funcs, iterations=iterations)


def benchsort(*funcs, iterations:Integral=100) -> dict[Callable, int]:
	data = Counter(repeatfunc(fastest_timer, iterations, *map(Timer, funcs)))
	return {timer.inner.__defaults__[0]:n for timer,n in data.most_common()}