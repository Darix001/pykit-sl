from itertools import tee
from collections.abc import Iterable, Callable, Iterator
from typing import Any


def item_map(func:Callable, iterable:Iterable, /) -> Iterator[tuple[Any, Any]]:
	it1, it2, = tee(iterable)
	return zip(it1, map(func, it2))


del Iterable, Callable, Iterator