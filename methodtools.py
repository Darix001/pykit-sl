import builtins, operator
from functools import update_wrapper, partial
from types import FunctionType, MethodType
from collections import UserList


basecode = UserList.__len__.__code__

func_args = operator.attrgetter('__code__', '__globals__', '__name__',
	'__defaults__', '__closure__')

ASSIGNMENTS = ('__module__', '__qualname__', '__doc__', '__annotations__',
	'__type_params__', '__kwdefaults__',)

COMPARE = ('__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__',)

BINARY = ('__add__', '__sub__', '__floordiv__', '__truediv__', '__mod__',
	'__mul__', '__matmul__', '__rshift__', '__lshift__', '__or__', '__and__',
	'__xor__',)

INPLACE = ('__iadd__', '__isub__', '__ifloordiv__', '__itruediv__', '__imod__',
	'__imul__', '__imatmul__', '__irshift__', '__ilshift__', '__ior__',
	'__iand__', '__ixor__',)

UNARY = ('__abs__', '__pos__', '__neg__', '__invert__',)


class unassigned_method:
	'''takes a method and assigns it the name of the given class variable'''
	__slots__ = 'factory'
	
	def __init__(self, factory, /):
		self.factory = factory

	def __set_name__(self, cls, name, /):
		add_method(cls, self.factory, name)


def add_method(cls, func, /, name=None, copy=None):
	if copy:
		func = func_copy(func)
	if not name:
		name = func.__name__
	else:
		func.__name__ = name
	func.__qualname__ = f"{cls.__qualname__}.{name}"
	func.__module__ = cls.__module__
	setattr(cls, name, func)
	return func


def func_copy(func, /):
	'''Create a shallow copy of a function'''
	return update_wrapper(FunctionType(*func_args(func)), func, ASSIGNMENTS)


class set_name(unassigned_method):
	__slots__ = ()

	def __set_name__(self, cls, name, /):
		add_method(cls, self.factory(name))


def dunder_method(attr, /, namespace=builtins.__dict__):
	code = basecode.replace(co_names=('_', attr))
	
	def factory(name, /):
		return FunctionType(code, {'_':namespace[name.strip('_')]}, name)
	
	factory.with_func = lambda func, /:FunctionType(code, {'_':func})
	return set_name(factory)


class operator_method(set_name):
	__slots__ = ('namespace', 'methods')

	def __init__(self, factory, /, namespace, methods):
		self.factory = partial(FunctionType, factory.__code__,
			factory.__globals__)
		self.namespace = namespace
		self.methods = methods

	def __set_name__(self, cls, name, /):
		methods = self.methods
		defs = zip(map(self.namespace.get, methods))
		for method in map(self.factory, methods, defs):
			add_method(cls, method)


def op_method(methods, /, namespace=operator.__dict__, rigth=None):
	if rigth:
		namespace = {namespace.get(key.replace('__', '__r', 1),
			namespace[key]) for key in methods}
	return partial(operator_method, methods=methods, namespace=namespace)


compare_method = op_method(COMPARE)

binary_method = op_method(BINARY)

inplace_method = op_method(INPLACE)

unary_method = op_method(UNARY)


del builtins, UserList