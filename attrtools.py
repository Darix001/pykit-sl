from __future__ import annotations
from itertools import chain, repeat
from operator import attrgetter
from . import methodtools


class Slots:
	'''Code Generator for classes with defined members.'''
	__match_args__ = __args__ = __slots__ = ()
	_new = classmethod(object.__new__)
	_new_method = classmethod(methodtools.add_method)

	def __init_subclass__(cls, /, add_repr=True, defs=None, init=True,
		abstract=False, starred=False, **kwargs):
		super().__init_subclass__(**kwargs)
		slots = cls.__slots__
		if abstract:
			cls.__match_args__ = slots
		elif slots:
			cls.__match_args__ = slots = cls.__match_args__ + slots
			namespace = vars(cls)			
			cls.__init = init = methodtools.initializer(slots, defs)
			cls.__values = property(attrgetter(*slots))
			add = cls._new_method

			if init and '__init__' not in namespace:
				add(init)

			if add_repr and '__repr__' not in namespace:
				string = f"({'=%r, '.join(slots)}=%r)"
		
				@add
				def __repr__(self, /):
					return self.__class__.__name__ + (string % self.__values)


	@methodtools.compare_method
	def __eq__(self, value, func, /):
		if self.__class__ is value.__class__:
			return func(self.__values, value.__values)
		return NotImplemented

	def __hash__(self, /):
		return hash(self.__values)

	def __repr__(self, /):
		return f'{self.__class__.__name__}{self.__values!r}'

	def __reduce__(self, /):
		return (type(self), self.__values)

	def __copy__(self, /):
		value = self._new()
		value.__init(*self.__values)
		return value

	property
	def __iterattrs(self, /):
		return map(getattr, repeat(self), self.__match_args__)

	def _replace(self, /, **data) -> Slots:
		args = map(data.pop, self.__match_args__, self.__iterattrs())
		self = super()._new()
		self.__init(*args)
		if not data:
			return self
		raise TypeError("Unexpected keyword argument: " + next(iter(data)))

	def _asdict(self, cls=dict, /) -> dict:
		return cls(zip(self.__match_args__, self.__iterattrs()))


def slotsfunc(func, /):
	return lambda self, /: func(*self.__values)