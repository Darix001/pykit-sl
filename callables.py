from collections.abc import Callable, MappingView, __loader__
from types import (MethodType as Bounder, SimpleNamespace as bounders,
	FunctionType)
from reprlib import Repr
import __future__

_SENTINEL = object()


def attrsetter(field_names:tuple[str], /, defaults:tuple = None) -> Callable:
	'''Returns an initializer with given attrs and default arguments'''
	if code := cache.get(size := len(field_names)):
		code = code.replace(co_names = field_names,
			co_varnames = ('self',) + field_names)
	else:
		data = ('\tself.%s=%%s\n' * size) % field_names
		data %= field_names
		cache[size] = code = compile(
			f"def __init__(self,/,{','.join(field_names)}):\n{data}",
			'<string>','exec').co_consts[0]
	return FunctionType(code, cache, '__init__', defaults)


def attrsetter1(field, default=_SENTINEL, /):
	defaults = (default,) if default is _SENTINEL else ()
	code = cache[1].replace(co_names = (field,), co_varnames = ('self', field))
	return FunctionType(code, cache, '__init__', defaults)
		


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


def incrementor(x:int = 1, /) -> Callable[(str,), str]:
    '''A callable that increments string chars x steps'''
    x = range(x, x+1114112)
    return lambda self,/: self.translate(x)


def revert_args(func, /):
	return lambda *args: func(args[::-1])


bounders = bounders()

Bounder = Bounder(Bounder, Bounder)

bounders.next, bounders.map, bounders.zip, bounders.iter = map(
	Bounder, (next, map, zip, iter,))

return_value = bounders.next(iter(b''))

method = compose(Bounder, property, doc='Converts any callable into a method.')

cache = {
	1:MappingView.__init__.__code__, 2:__loader__.__init__.__code__,
	3:__future__._Feature.__init__.__code__,
	13:Repr.__init__.__code__.replace(co_kwonlyargcount=0, co_argcount=14)}

del Callable, MappingView, __future__, __loader__, Repr