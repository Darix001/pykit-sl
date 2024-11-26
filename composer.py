from collections.abc import Callable
from types import (MethodType as Bounder, SimpleNamespace as bounders,
	FunctionType)


def compose(*args:tuple[Callable], doc = None) -> Callable:
	'''Combines passed functions into one single callable.'''
	def func(self, /):
		for func in args:
			self = func(self)
		return self
	func.__doc__ = doc
	return func


def compose_n(obj:Callable, x:int, /, *, doc = None) -> Callable:
	'''Same as compose(*(func,)*x)'''
	x = range(x)
	def func(self, /):
		for _ in x:
			self = obj(self)
		return self
	func.__doc__ = doc
	return func


def class_compose(*args:tuple[Callable], doc = None) -> classmethod:
	'''wraps a classmethod around composed functions'''
	@classmethod
	def func(cls, self, /):
		for func in args:
			self = func(self)
		return cls(self)
	func.__doc__ = doc
	return func


def revert_args(func, /):
	return lambda *args: func(args[::-1])