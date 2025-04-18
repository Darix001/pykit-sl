import __future__, builtins, operator

from reprlib import Repr
from typing import Any
from operator import attrgetter
from collections import UserList
from types import FunctionType, CodeType
from functools import update_wrapper, partial, wraps
from collections.abc import (MappingView, __loader__, Callable, Iterable,
	Mapping)


basecode = UserList.__len__.__code__
 
func_args = attrgetter('__code__', '__globals__', '__name__',
	'__defaults__', '__closure__')

ASSIGNMENTS = ('__module__', '__qualname__', '__doc__', '__annotations__',
	'__type_params__', '__kwdefaults__',)


class Base:
	def __init__(self, func, /):
		self.func = func
		

class multigetter(Base):
	'''A getter for both classes and instances'''

	def __get__(self, instance, owner=None, /):
		return self.func(owner if instance is None else instance)


class unassigned_method(Base):
	'''takes a method and assigns it the name of the given class variable'''

	def __set_name__(self, cls, name, /):
		add_method(cls, self.func, name)


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


@classmethod
def method_adder(cls, method=None, /, **kwargs):
	if method is not None:
		add_method(cls, method)
	else:
		return partial(add_method, **kwargs)


def to_method(func:Callable, /) -> Callable:
	return lambda self, /: func(self)


def func_copy(func, /) -> Callable:
	'''Create a shallow copy of a function'''
	return update_wrapper(FunctionType(*func_args(func)), func, ASSIGNMENTS)


def newglobals(func, globals, /):
	function = FunctionType(func.__code__, globals,
		func.__name__, func.__closure__, func.__defaults__)
	return update_wrapper(function, func)


class setname_factory(unassigned_method):
	'''Given a function, wich assigned multiple names on a class, 
	passes the name to the function and attachs the new resulting functions
	to the assigned class variable names.
	
	Example:
	@set_name
	def some_method(name, /):
		return lambda self: self.method(name)

	class A:
		word = other_word = variable_name = some_method

	print(A.word)
	#prints <A.word function at ...>

	'''

	def __set_name__(self, cls, name, /):
		add_method(cls, self.func(name), name)


class set_name(unassigned_method):
	'''Given a function, wich assigned multiple names on a class, 
	passes the name to the function and assigns the result of the
	function call to the variable name class.
	
	Example:

	class A:
		word = other_word = variable_name = set_name(str.upper)

	print(A.word)
	#prints WORD

	'''
	def __set_name__(self, cls, name, /):
		setattr(cls, name, self.func(name))
		

getinitcode = attrgetter('__init__.__code__')

inits = dict(
	enumerate(
		map(getinitcode, (MappingView, __loader__, __future__._Feature)),
		start=1)
	)
inits[13] = Repr.__init__.__code__.replace(co_kwonlyargcount=0, co_argcount=14)


# new_method_code = attrgetter('__new__.__code__')

# classes = map(vars(inspect).get,
# 	('Arguments', 'ClosureVars','_Traceback', '_FrameInfo', 'FullArgSpec')
# 	)

# new_method_codes = dict(
# 	enumerate(map(new_method_code, classes), start=3)
# 	)
# new_method_codes[9] = dis._Instruction.__new__.__code__


def constructor(cache:dict[int, CodeType], methodname:str,
	first_argument_name:str='self') -> Callable:

	initial_string = f"def {methodname}(self,/,"
	first_argument_name = first_argument_name,

	def decorator(func:Callable, /) -> Callable:
		def function(namespace, slots:bool=False) -> type|Mapping:
			if callable(namespace):
				namespace = vars(namespace)

			annotations = namespace['__annotations__']
			field_names = annotations.keys()
			classattrs = namespace.keys()
	
			if slots and '__slots__' not in namespace:
				namespace['__slots__'] = field_names
			
			if field_names < classattrs:
				func = namespace.pop if slots else namespace.get
				defs = tuple(map(func, field_names))

			if code := cache.get(size := len(field_names)):
				field_names = tuple(field_names)
				code = code.replace(
					co_names = field_names,
					co_varnames = first_argument_name + field_names,
					co_filename='<string>',
					co_name=methodname,
					co_qualname=f"{namespace['__qualname__']}.{methodname}")

			else:
				join = ','.join(field_names)
				string = f"{initial_string}{join}):\n"
				string = func(string, size, field_names, defs, join)
				cache[size] = code = compile(string, '<string>', 'exec'
					).co_consts[0]

			namespace[methodname] = func = FunctionType(code, {},
				methodname, defs)
			func.__annotations__ = annotations.copy()

			return namespace

		return function

	return decorator

			
@constructor(inits, '__init__')
def init(initial_string, size, field_names, defs, join, /):
	'''Returns an initializer with given attrs and default arguments'''
	data = ('\tself.%s=%%s\n' * size) % field_names
	data %= field_names
	return initial_string + data
	

@constructor({}, '__new__', 'cls')
def tuplenew(initial_string, size, field_names, defs, join, /):
	return f"{initial_string}\ttuple.__new__(cls, ({join}))"


def set_coname(func, /, dec=None):
	args = [*func_args(func)]
	index = (co_names := (code := args[0]).co_names).index('_')
	co_names = [*co_names]
	wrapper = wraps(func)
	
	def decorator(name, /):
		co_names[index] = name
		args[0] = code.replace(co_names=tuple(co_names))
		return wrapper(FunctionType(*args))
	
	return decorator


def copy_fromcls(cls, /):
	@setname_factory
	def factory(name, /):
		return func_copy(getattr(cls, name))


def dunder_method(module, strip=None):
	
	def decorator(func, /):
	
		@setname_factory
		def function(name, /):
			return func(getattr(module, name.strip('_') if strip else name))
	
		return function
	
	return decorator


operator_method = dunder_method(operator)

builtin_method = dunder_method(builtins, strip=True)


# class namespace:
# 	'''Class that accepts dynamic attributes.
# 	Similiar to sys.namespace, but this class keeps the key shared dict PEP
# 	of python.'''
# 	def __init__(self, /, **kwargs):
# 		for attr, value in kwargs.items():
# 			setattr(self, attr, value)


# # def dataclass(cls=None, /, *, init=True, copy=True):
# # 	if cls is None:
# # 		del cls
# # 		return partial(dataclass, **locals())
	
# # 	namespace = vars(cls)
# # 	namespace_names = namespace.keys()
# # 	annotations = get_annotations(cls)
	
# # 	fields = namespace.get('__slots__') or annotations.keys():
	
# # 	if namespace_names.isdisjoint(fields):
# # 		defs = None
	
# # 	elif namespace_names > fields:
# # 		defs = tuple(map(namespace.get, fields))

# # 	if init and '__init__' not in namespace:
# # 		fields = tuple(fields)
# # 		add_method(cls, init := initializer(fields, defs))
# # 		init.__annotations__ = annotations.copy()
		
# # 	if copy:
# # 		getattrs = attrgetter(*fields)
		
# # 		def __copy__(self, /):
# # 			type(self)(*getattrs(self))
		
# # 		add_method(cls, __copy__)



# # def slots(data, /):
# # 	if fields := data.get('__annotations__'):
# # 		fields = fields.keys()
# # 	data['__slots__'] = fields


del (builtins, UserList, Repr, Any, MappingView, Callable, Mapping, Iterable,
	attrgetter, operator)