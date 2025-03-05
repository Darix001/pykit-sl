from . import methodtools
from dataclasses import dataclass, _replace


@dataclass
class Full(BaseArray):
	_shape:list[int]
	fill_value:Any
	
	base = None
	_replace = replace

	def __getitem__(self, index, /):
		shape = [*self._shape]
		if (index_type := type(index)) is tuple:
			if not index:
				return self
			
			if len(index) > len(shape):
				raise IndexError(
					"too many indices for full: full is"
					f"{ndim}-dimensional, but 2 were indexed")

			else:
				try:
					indices = iter(index)
					for axis, dim in enumerate(map(range, shape)):
						for index in indices:
							if type(index := dim[index]) is range:
								shape[0] = len(dim)
							else:
								del shape[0]
								break

				except IndexError as e:
					raise TypeError(
						f"index {index} is out of bounds for axis"
						f"{axis} with size {shape[0]}"
						) from None
		if index is not ...:
			if shape := self._shape:
				axis = 0
				first_dim = range(shape[0])

				try:
					index = first_dim[index]
				except IndexError:
					raise IndexError(
						f"Index {index} is out of bounds for axis"
						f" 0 with size {shape[0]}."
							) from None

				if index_type is slice:
					shape[0] = len(first)

				else:
					axis = 1
					del shape[0]

			else:
				try:
					shape[index]#If the index is not slice an error will be raised.
				except IndexError:
					raise IndexError(
						f"index {index} is out of bounds for axis 0 with size 0"
						) from None
			
				'''Since we confirm that the index is slice and self is empty,
				simply return self as ther is nothing to slice.'''
		return self.copy()

		return self._getitem(axis, shape)


	@property
	def T(self, /):
		return self._replace(shape=self._shape[::-1])

	ndims, shape = map(property, map(shape_method.with_func, (len, tuple)))

	def copy(self, /):
		return self._replace(_shape=[*self._shape])

	def pop_axis(self, axis:int, /) -> tuple[list[range], int,]:
		shape = [*self._shape]
		size //= (axis := shape.pop(axis))
		return shape, size, axis

	def __array__(self, /):
		from numpy import full
		return numpy.full(self._shape, self.fill_value)

	def __iter__(self, /):
		if not (shape := self._shape):
			return EMPTY_ITERATOR

		repeater = repeat(self.fill_value, shape[0])

		if len(shape) == 1:
			return repeater

		else:
			return map(type(self), repeat(shape[1:]), repeater)

	__reversed__ = __iter__

	def __round__(self, decimal_places=None, /):
		self._replace(fill_value=round(self.fill_value, decimal_places))

	def astype(self, dtype, /):
		self._replace(fill_value=dtype(self.fill_value))

	__abs__ = methodtools.unary_method(astype)

	def _getitem(self, axis:int, shape:list[range], /):
		return self._replace(_shape=shape) if shape else self.fill_value

	@property
	def dtype(self, /) -> type:
		return type(self.fill_value)

	@property
	def flat(self, /) -> Iterator[Any]:
		return irepeat(self.fill_value, self.size)
		
	imag = real = numerator = denominator = fill_value_property

	def conjugate(self, /):
		return self._replace(fill_value=self.fill_value.conjugate())

	def fill(self, value, /):
		object.__setattr__(self, 'fill_value', value)

	def flatten(self, /):
		return self._replace(shape=(self.size,))

	ravel = flatten

	def transpose(self, axis=None, /):
		if axis is None:
			return self.T
		else:
			pass

	def moveaxes(self, axis1, axis2, /):
		shape = self._shape
		shape[axis1], shape[axis2] = shape[axis2], shape[axis1]

	def swapaxes(self, axis1, axis2, /):
		self = self.copy()
		self.moveaxes(axis1, axis2)
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
		shape = [*shape]
		if self.size != (size := prod(shape)):
			raise ValueError(
				f"cannot reshape array of size {size} into shape {shape}")
		return self._replace(shape=shape, size=size)

	@reshape_func
	def resize(self, shape):
		_shape = self.shape
		_shape[:] = shape
		object.__setattr__(self, 'size', prod(_shape))

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
		return self._replace(shape=shape, size=size)

	def math_method(func, /):
		@wraps(func)
		def function(self, axis=None, /):
			return func(self, axis, axis is not None, self.fill_value)
		return function

	@math_method
	def prod(self, axis, along_axis, value, /):
		if along_axis:
			shape, size, axis = self.pop_axis(axis)
			value **= axis
			return self._replace(shape=shape, size=size, value=value)
		else:
			return value ** self.size

	@math_method
	def sum(self, axis, along_axis, value, /):
		if along_axis:
			shape, size, axis = self.pop_axis(axis)
			value *= axis
			return self._replace(shape=shape, size=size, value=value)
		else:
			return self.size * value

	@math_method
	def cumprod(self, axis, along_axis, value, /):
		if along_axis:
			pass
		else:
			pass

	@math_method
	def cumsum(self, axis, along_axis, value, /):
		if along_axis:
			pass
		else:
			size = self.size
			end = value * (size + 1)
			return Arange(size, (size,), value, end, value)

	def axis_func(self, /, value, axis):
		if axis is not None:
			return value
		else:
			shape, size, axis = self.pop_axis(axis)
			return self._replace(shape=shape, size=size, value=value)

	def std(self, axis=None, /):
		if shape := self.shape:
			if 0 in shape:
				return nan
		self.axis_func(.0, axis)

	var = std


	@methodtools.set_name
	def argmax(name, /):
		msg = f"attempt to get {name} of an empty sequence"
		def function(self, axis=None, ):
			if self.size:
				return self.axis_func(0, axis)
			else:
				raise ValueError(msg)
		return function


	argmin = argmax


	@methodtools.set_name
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