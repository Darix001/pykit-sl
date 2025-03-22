from collections.abc import Callable
from types import FunctionType
from functools import wraps


def composer(func) -> Callable:
	@wraps(func)
	def function(*args:tuple[Callable], doc=None) -> Callable:
		(composed_func := func(args)).__doc__ = doc
		return composed_func
	return function


@composer
def compose(args, /):
	first, *funcs = args
	def func(*args, **kwargs):
		obj = first(*args, **kwargs)
		for func in funcs:
			obj = func(obj)
		return obj
	return func

@composer
def simple_compose(args, /) -> Callable:
	'''Combines passed functions into one single callable.'''
	def func(self, /):
		for func in args:
			self = func(self)
		return self
	return func

@composer
def compose_n(args) -> Callable:
	'''compose_n(func1, func2, func3..., x, doc=None)'''
	*functions, x = kwargs
	x = range(x)
	def func(self, /):
		for _ in x:
			for function in functions:
				self = function(self)
		return self
	func.__doc__ = doc
	return func


@composer
def simple_class_compose(*args:tuple[Callable], doc = None) -> classmethod:
	'''wraps a classmethod around composed functions'''
	args = args[::-1]
	@classmethod
	def func(cls, self, /):
		for func in args:
			self = func(self)
		return cls(self)
	func.__doc__ = doc
	return func


def bicompose(f1:Callable, f2:Callable):
	return lambda obj, /: f1(f2(obj))