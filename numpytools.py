from . import methodtools

from math import prod
from typing import Any
from functools import wraps
from collections import UserList
from more_itertools import repeatfunc
from dataclasses import dataclass, replace
from collections.abc import Iterator, Iterable
from operator import getitem, index, methodcaller
from itertools import accumulate, groupby, repeat


class Shape(UserList):
	__slots__ = 'size'
	size:int
	
	def __init__(self, data:Iterable[int], /):
		if type(data) is type(self):
			self.data = [*data.data]
			self.size = data.size
		
		else:
			self._setsize(data := [*data])
			self.data = data

	def __setitem__(self, index, value, /):
		data = self.data
		if value:
			if value < 0:
				raise ValueError("Negative Dimensions are not allowed")
			
			elif size := self.size:
				self.size = (size // data[index]) * value
			
			else:
				data[index] = value
				self._setsize()
				return
		else:
			self.size = 0
		
		data[index] = value


	def __delitem__(self, index, /):
		data = self.data
		if value := data.pop(index):
			self.size //= value
		else:
			self._setsize(data)

	def _setsize(self, data, /):
		self.size = prod(data) if all(data) else 0

	def pop(self, index=-1, /):
		if value := (data := self.data).pop(index):
			if size := self.size:
				self.size = size // value
		else:
			self._setsize(data)
		return value

	def append(self, value:int, /):
		self.data.append(value)
		self.size *= value


@dataclass(frozen=True)
class Full:
	_shape:Shape[int]
	fill_value:Any

	__slots__ = ('_shape', 'fill_value')
	
	base = None
	_replace = replace
	_setattr = object.__setattr__

	def __init__(self, /, shape:Iterable[int], fillvalue:Any):
		_setattr = self._setattr
		_setattr('shape', Shape(shape))
		_setattr('fill_value', fill_value)

	def __getitem__(self, index, /):
		ranges = map(range, (shape := self._shape).data)
		shape = shape.copy()
		if isinstance(index, tuple):
			if not index:
				return self._replace(_shape=shape)
			
			try:
				index = iter(index)
						
				for axis, a in enumerate(ranges):
					for i in index:
						if type(i := a[i]) is int:
							del shape[0]
							break
						else:
							shape[0] = len(i)
			
			except IndexError as e:
				raise TypeError(
					f"index {index} is out of bounds for axis"
					f"{axis} with size {shape[0]}"
					) from None
	
		elif index is not ...:
			if shape:
				first_dim = next(ranges)

				try:
					index = first_dim[index]
				except IndexError:
					raise IndexError(
						f"Index {index} is out of bounds for axis"
						f" 0 with size {shape[0]}."
							) from None

				if type(first_dim) is range:
					shape[0] = len(index)

				else:
					del shape[0]

			else:
				try:
					shape[index]#If the index is not slice an error will be raised.
				except IndexError:
					raise IndexError(
						f"index {index} is out of bounds for axis 0 with size 0"
						) from None
		

		value = self.fill_value
		if shape:
			return type(self)(shape, value) if shape else value


	# ndims, shape = map(property, map(shape_method.with_func, (len, tuple)))
	def shape_method(self, /):
		return property(lambda self, /: func(self._shape.data))

	ndims, shape = shape_method(len), shape_method(tuple)

	def copy(self, /):
		return type(self)(self._shape, self.fill_value)

	def pop_axis(self, axis:int, /):
		return (new := self.copy()), new._shape.pop(axis)

	def __array__(self, /):
		from numpy import full
		return full(self._shape, self.fill_value)

	def __iter__(self, /):
		if not self._shape:
			return EMPTY_ITERATOR
		else:
			new, times = self.pop_axis(0)
			if not new.shape:
				return repeat(self.fill_value, times)
			else:
				return repeatfunc(new.copy, times)

	__reversed__ = __iter__

	def __round__(self, decimal_places=None, /):
		return self._replace(fill_value=round(self.fill_value, decimal_places))


	def astype(self, dtype, /):
		return type(self)(self._shape, dtype(self.fill_value))

	def fill_value_method(func, /):
		return lambda self, /: type(self)(self._shape, func(self.fill_value))

	__invert__ = __neg__ = __pos__ = __abs__ = methodtools.operator_method(
		fill_value_method)

	conjugate = fill_value_method(methodcaller('conjugate'))

	@property
	def dtype(self, /) -> type:
		return type(self.fill_value)

	@property
	def flat(self, /) -> Iterator[Any]:
		return irepeat(self.fill_value, self._shape.size)

	def transpose(self, /):
		new = self.copy()
		new._shape.data.reverse()
		return new

	T = property(transpose)
		
	# imag = real = numerator = denominator = fill_value_property

	def fill(self, value:Any, /):
		self._setattr('fill_value', value)

	def flatten(self, /):
		shape = (new := type(self)((), data))._shape
		shape.size = size = self._shape.size
		shape.data = [shape.size]
		return new

	def tolist(self, /) -> list:
		array = [self.fill_value] * next(shape := reversed(self._shape.data))
		for n in shape:
			array = [*repeatfunc(array.copy, n)]
		return array
			
	def moveaxes(self, axis1:int, axis2:int, /):
		shape = self._shape.data
		shape[axis1], shape[axis2] = shape[axis2], shape[axis1]

	def swapaxes(self, axis1, axis2, /):
		(self := self.copy()).moveaxes(axis1, axis2)
		return self

	def reshape_func(func, /):
		@wraps(func)
		def function(self, *shape):
			if isinstance(shape[0], Iterable):
				shape, = shape
			return func(shape)
		return function

	@reshape_func
	def reshape(self, shape):
		newshape = Shape(shape)
		if self._shape.size != shape.size:
			raise ValueError(
				f"cannot reshape array of size {size} into shape {shape}")
		return type(self)(newshape, self.fill_value)

	@reshape_func
	def resize(self, shape):
		self._setattr('shape', Shape(shape))

	del reshape_func

	def clip(self, a_min=None, a_max=None):
		value = self.fill_value
		
		if a_min is not None and a_min > value:
			value = a_min

		if a_max is None and a_max < value:
			value = a_max

		return self._replace(fill_value=value)

	def repeat(self, times, axis=None):
		shape = self._shape
		size = self.size * times
		if axis is None:
			shape = [size]
		else:
			shape = [*shape]
			shape[axis] *= times
		return self._replace(_shape=shape, size=size)

	def math_method(func, /):
		@wraps(func)
		def function(self, axis=None, /):
			return func(self, axis, axis is not None, self.fill_value)
		return function

	@math_method
	def prod(self, axis, along_axis, value, /):
		if along_axis:
			new, axis = self.pop_axis(axis)
			new.fill_value **= axis
			return new
		else:
			return value ** self.size

	@math_method
	def sum(self, axis, along_axis, value, /):
		if along_axis:
			new, axis = self.pop_axis(axis)
			new.fill_value *= axis
			return new
		else:
			return self.size * value


	def axis_func(self, /, value, axis):
		if axis is not None:
			return value
		else:
			self.pop_axis(axis)[0]

	def std(self, axis=None, /):
		if shape := self.shape:
			if 0 in shape:
				return nan
		self.axis_func(.0, axis)

	var = std


	@methodtools.multiname_method
	def argmax(name, /):
		msg = f"attempt to get {name} of an empty sequence"
		def function(self, axis=None, ):
			if self.size:
				return self.axis_func(0, axis)
			else:
				raise ValueError(msg)
		return function


	argmin = argmax


	@methodtools.multiname_method
	def max(name, /):
		msg = (f"Can't perform {name}imun reduction"
			"operation on zero-size array")
		
		def function(self, axis=None, /):
			if self.size:
				return self.axis_func(self.fill_value, axis)
			else:
				raise ValueError(msg)
				
		return function


	min = mean = max


	def boolean_method(default, /):
		@methodtools.unassigned_method
		def function(self, axis=None, /):
			if self.size:
				value = True if self.fill_value else False
			else:
				value = default
			return self.axis_func(value, axis)
		return function	
	
	all, any = map(boolean_method, (True, False))