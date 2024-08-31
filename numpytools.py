import numpy, sys, seqtools
from operator import index

class arange:
	__slots__ = 'data'
	
	def __init__(self, /, *args, dtype=int):
		start, stop, step = seqtools.args(slice(*args))
		if start is None:
			start = 0
		if step == 0:
			raise ValueError("arange step cannot be zero.")
		self.data = data = numpy.array((start, stop, step), dtype)

	def __getitem__(self, key, /):
		data = self.data
		ndims = len(shape := self.shape)
		
		match type(key):
			case list:
				if all(isinstance(k, bool) for k in key):
					return acompress(self, key)
				else:
					return aindexed(self, key)

			case tuple:
				if key:
					pass

			case bool:
				return self._replace(data=data, shape=(+key, *self.shape))

			case _:
				if ndims == 1:
					value = (data[0] + key) * data[2]
					if key > data:
						raise IndexError(key)
					return value