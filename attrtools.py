from __future__ import annotations
from operator import attrgetter
from . import callables, methodtools

class Slots:
	'''Code Generator for classes with defined members.'''	

	__slots__ = __match_args__ = __args__ = ()

	_new_method = classmethod(methodtools.add_method)

	__new = classmethod(object.__new__)

	def __init_subclass__(cls, /, *,  defaults=None, frozen=None, repr=None):
		if field_names := cls.__slots__:
			cls.__match_args__ = field_names = cls.__match_args__ + field_names
			module, qualname =  cls.__module__, cls.__qualname__

			cls.__init = init = callables.attrsetter(field_names, defaults)
			add = cls._new_method

			if '__init__' not in (data := cls.__dict__):
				add(init)

			if repr and '__repr__' not in data:
				string = f"({'=%r, '.join(field_names)}=%r)"
				@add
				def __repr__(self, /):
					return self.__class__.__name__ + (string % self.__args__)

			cls.__args__ = property(attrgetter(*field_names))

		elif defaults:
			init = add(cls.__init__, copy=True)
			init.__defaults__ = defaults

		if frozen:
			@add
			def __setattr__(self, name, value, /):
				if not getattr(self, '__args__', None):
					return super(cls, self).__setattr__(name, value)
				raise AttributeError(f'Attribute {name} is read-only')

			@add
			def __delattr__(self, name, /):
				raise AttributeError(f'Attribute {name} is read-only')


	@methodtools.compare_method
	def __eq__(self, value, func, /):
		if self.__class__ is value.__class__:
			return func(self.__args__, value.__args__)
		return NotImplemented

	def __hash__(self, /):
		return hash(self.__args__)

	def __repr__(self, /):
		return self.__class__.__name__ + f'{self.__args__!r}'

	__reduce__ = callables.method(attrgetter('__class__', '__args__'))

	__sizeof__ = property(attrgetter('__args__.__sizeof__'))

	def __copy__(self, /):
		value = self.__new()
		value.__init(*self.__args__)
		return value

	def _replace(self, /, **data) -> Slots:
		args = map(data.pop, self.__match_args__, self.__args__)
		self = self.__new()
		self.__init(*args)
		if not data:
			return self
		raise TypeError("Unexpected keyword argument: " + next(iter(data)))

	def _asdict(self, cls=dict, /) -> dict:
		return cls(zip(self.__match_args__, self.__args__))

class Slot:
	__slots__ = __key = '__value'
	
	def __init_subclass__(cls, /, *, default=None, frozen=None, attrname=None):
		add = cls._new_method
		
		if attrname:
			setattr(cls, attrname, self.__value)
			cls.__key  = attrname
		
		if default:
			add(callables.attrsetter((cls.__key,), (default,)))
		
		if frozen:
			@add
			def __setattr__(self, attr, value, /):
				if hasattr(self, attr):
					raise AttributeError(f'Attribute {attr} is read-only')
				raise AttributeError(f"""'object' {self.__class__.__name__} has
					no attribute named '{attr}'""")
	

	def __repr__(self, /):
		return f"{self.__class__.__name__}({self.__value!r})"

	def __hash__(self, /):
		return hash((self.__class__, self.__value))

	@methodtools.compare_method
	def __eq__(self, obj, func, /):
		if self.__class__ is obj.__class__:
			return func(self.__value, obj.__value)
		return NotImplemented

	__reduce__ = callables.method(attrgetter('__class__', '__value'))

	__sizeof__ = property(attrgetter('__value.__sizeof__'))

	def __copy__(self, /):
		return self.__class__(self.__value)

	def _asdict(self, /):
		return {self.__key:self.__value}


def slotsfunc(func, /):
	return lambda self, /: func(*self.__args__)


Slots.__init__ = property(callables.method(Slot._Slot__value.__set__))