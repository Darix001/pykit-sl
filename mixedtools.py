from typing import Any
from types import MethodType as partializer
from itertools import repeat
from functools import partial
from collections.abc import Callable


partializer = partializer(partializer, partializer)


def revert_args(func:Callable, /) -> Callable:
	return lambda *args: func(args[::-1])


def constfunc(value:Any, /) -> Callable:
	return repeat(value).__next__


class partialt(partial):
	__slots__ = ()

	@classmethod
	def add_method_call(cls, name:str, /):
		setattr(cls, name, cls.__call__)



del Callable, Any, partial