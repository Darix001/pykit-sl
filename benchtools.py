from functools import partial
from collections import Counter
from timeit import Timer, timeit
from collections.abc import Callable
from more_itertools import repeatfunc


fastestimer = partial(min, key=Timer.timeit)

fastestfunc = partial(min, key=timeit)


def ibenchsort(*funcs, runs:int=100) -> dict[Callable, int]:
	'''Benchsort the functions every "runs" times'''
	data = Counter(repeatfunc(fastesttimer, runs, *map(Timer, funcs)))
	return {timer.inner.__defaults__[0]:n for timer,n in data.most_common()}

	
def benchsort(*funcs) -> list[Callable]:
	return sorted(funcs, key=timeit)