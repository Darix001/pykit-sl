'''This library is directly inspired by itertools, as the idea of conceiving a library
for dealing with seqtools in a more efficient way.'''
from __future__ import annotations

import itertools as it, math, operator as op


from typing import Any
from sys import maxsize
from numbers import Number
from types import MethodType
from more_itertools import locate
from functools import wraps, update_wrapper as wrap
from collections import deque, UserDict, UserList, Counter
from dataclasses import dataclass, replace, make_dataclass

from collections.abc import Sequence, Callable, Iterator, Iterable, Generator

from .mixedtools import partializer
from .composetools import bicompose


div_index = {0, -1}.__contains__

cycle = bicompose(from_iterable := it.chain.from_iterable,
	irepeat := it.repeat)

mapper = partializer(map)

frozen_dataclass = dataclass(frozen=True)


MAP = [it.repeat, from_iterable, reversed, it.islice,
op.itemgetter(slice(None, None, -1)), op.getitem,
op.floordiv, op.attrgetter('stop'), len, op.contains, op.countOf]

MAP[:] = map(mapper, MAP)

CC_MAP = MAP[-2:]
del MAP[-2:]
get_sizes = MAP.pop()


R_NONE = irepeat(None)

OPINT = int | None

ITII = Iterator[tuple[int, int]]

TS = tuple[Sequence]

NWISE_ITER = {1:zip, 2:it.pairwise}

MAXSIZE_RANGE = range(maxsize)

SENTINEL = object()


def slicer(func:Callable, /) -> Callable:
	'''Decorator for functions wich accepts one argument and range arguments'''
	def function(obj, /, *args):
		return func(obj, slice(*args))
	
	name = func.__name__
	function.__doc__ = f'{name}(obj, stop)\n{name}(obj, start, stop[, step])'
	
	return function


def efficient_nwise(iterable:Iterable, n:int) -> Generator[deque]:
	'''Yields efficients nwise views of the given iterable re-using a
	collections.deque object.'''
	data = deque(it.islice(iterable := iter(iterable), n - 1), n)
	for _ in map(data.append, iterable):
		return data


def getitems(data:Sequence, items, /) -> Iterator:
	'''fetchs and Yields each item of the data object.'''
	return MAP[5](irepeat(data), items)


def checker(cls, /):
	'''Creates a Check method for SubSequence subclasses'''
	return lambda self, obj, /: type(obj) is cls and len(obj) == self.r


def check_step(step:int, /):
	if not step:
		raise TypeError("Step Argument must not be zero.")


def datamethod(func:Callable, /) -> Callable:
	return lambda self,/: func(self.data)


def zipbool(func:Callable, /) -> Callable:
	return lambda self,/: True if (data := self.data) and func(data) else False


def calcsize(func:Callable, /) -> Callable:
	return lambda self, /: func(get_sizes(self.data))


def comb_len(cls, /) -> type:
	'''Decorator for combinations and permutation classes that computes their get_sizes based
	on their respective math function.'''
	func = getattr(math, cls.__name__[:4])
	cls.__len__ = lambda self, /: func(len(self.data), self.r)
	return cls


def efficient_slice(data:Sequence, slicer:slice):
	if isinstance(data, Slice):
		return data._replace(r=data.r[slicer])

	elif not slicer.start and slicer.stop is None:
		match slicer.step:
			case None|1:
				return SequenceView(data)

			case -1:
				return ReverseView(data)

	return efficient_slice(data, range(len(data))[slicer])
		

islice = slicer(efficient_slice)


def get(data:Sequence, index:int, default:Any=None, /):
	'''Return the value for key if key is in the sequence, else default.'''
	try:
		return data[index]
	except IndexError:
		return default


def all_equals(data:Sequence, /) -> bool:
	'''Returns True if all items on sequence are equals'''
	return data.count(data[0]) == len(data)


# def sub(data:Sequence, values:Sequence, start:int=0, stop:OPINT=None, /):
# 	first = values[0]
# 	values = Slice(values, range(diff, len()))
# 	while diff:
# 		index = start = data.index(first, start, stop)
# 		for start, value in enumerate(values, start + diff):
# 			if diff := value != data[start]:
# 				start += diff
# 				break
# 	return index


class BaseSequence(Sequence):
	'''Base class for all classes in this module.'''
	__slots__ = ()
	iterfunc = None
	_replace = replace
	_setattr = object.__setattr__
	
	def __init_subclass__(cls, /):
		if (factory := cls.iterfunc) is not None:
			cls.__iter__, cls.__reversed__ = map(factory, (None, True))
			del cls.iterfunc
	
	def value_error(self, value, /):
		raise ValueError(f"{value!r} not in {type(self).__name__}")

	def index_error(self, /):
		raise IndexError(f"{type(self).__name__} index out of range.")

		
@dataclass(frozen=True, order=True)
class SequenceView(BaseSequence):
	'''A view over a new sequence.'''
	__slots__ = 'data'
	data:Sequence
	
	index, count = UserList.index, UserList.count
	
	__len__, __contains__ = UserList.__len__, UserList.__contains__
	
	__iter__ = UserDict.__iter__
	
	def __getitem__(self, index, /):
		data = self.data
		if type(index) is slice:
			if len(r := range(n := len(data))[index]) == n:
				if r.step > 0:
					return self
				else:
					return ReverseView(data)
			return Slice(data, r)
		else:
			return data[index]

		
	__reversed__ = datamethod(reversed)

	__bool__ = datamethod(bool)

	def __mul__(self, n, /):
		return mul(self.data, n)


class ReverseView(SequenceView):
	'''Reversed View of a sequence data'''
	__slots__ = ()
	__reversed__, __iter__ = SequenceView.__iter__, SequenceView.__reversed__

	def __getitem__(self, index, /):
		data = self.data
		if type(index) is slice:
			if len(r := range(n := len(data) -1, -1, -1)[index]) == n:
				if step < 0:
					return SequenceView(data)
				else:
					return self
			return Slice(data, r)
		else:
			return data[~index]

	def index(self, value:Any, start:int=0, stop:OPINT=None, /) -> int:
		n = len(data := self.data)
		getindex = data.index
		if not start and stop is None:
			return ~getindex(value)
		else:
			return ~getindex(value, ~stop + n, ~start + n)


class Size(SequenceView):
	'''Base Class for sequence wrappers that transform their sequence size.'''
	__slots__ = ()

	def __bool__(self, /):
		return True if self.data and self.r else False


class BaseIndexed(BaseSequence):
	__slots__ = ()

	def __getitem__(self, index, /):
		if type(r := self.r[index]) is int:
			return self._getitem(r)
		else:
			return self._getslice(r)

	def __contains__(self, value, /):
		if indices := self.r:
			return self._contains(value, indices)
		else:
			return False

	def index(self, value, start:int=0, stop:OPINT=None, /):
		if indices := self.r[start:stop]:
			return self._index(value, indices)
		else:
			self.value_error(value)

	def count(self, value, start:int=0, stop:OPINT=None, /):
		if indices := self.r[start:stop]:
			return self._count(value, indices)
		else:
			return 0


@frozen_dataclass
class Ranged(BaseIndexed):
	__slots__ = 'r'
	r:range


@frozen_dataclass
class Indexed(Size, BaseIndexed):
	r:Sequence[int]

	def __len__(self, /):
		return len(self.r) if self.data else 0
	
	def iterfunc(reverse, /):
		def __iter__(self, /):
			indices = self.r
			return getitems(data, reversed(indices) if reverse else indices)
		return __iter__

	def _getitem(self, index:int, /):
		return self.data[index]

	def _getslice(self, r:Sequence[int], /):
		return type(self)(self.data, r)

	def _index(self, obj, indices, /):
		return indices.index(self.data.index(obj))

	def _count(self, obj, indices, /):
		return sum(map(Counter(indices).get, locate(self.data, obj)))

	def unpack(self, /) -> Iterator:
		return getitems(self.data, self.r)


class Slice(Indexed):
	__slots__ = ()

	def __reversed__(self, /):
		size = len(data := self.data)
		if start := (r := self.r).start:
			return getitems(data, r)
		return it.islice(it.chain((None,), reversed(data)),
			size - start, size - self.stop, abs(self.step))
	
	def __iter__(self, /):
		gen = iter(data := self.data)
		stop, step = r.stop, r.step
		if start := r.start:
			try:
				gen.__setstate__(start)
			except AttributeError:
				return getitems(data, r)
			else:
				stop -= start
				start = 0
		return it.islice(gen, start, stop, step)

	def _contains(self, value, indices, /):
		try:
			return self.data.index(value,
				indices.start, indices.stop) in indices
		except ValueError:
			pass
		except TypeError:
			return value in iter(self)

	#bool and integer functions decorator
	def boolen(func, FALSIES={'__bool__':False, '__len__':0}, /):
		value = FALSIES[func.__name__]
		@wraps(func)
		def function(self, /):
			if (data := self.data) and (r := self.r):
				return func(data, r)
			return value
		return function

	@boolen
	def __bool__(data, r, /):
		return start < min(len(data), r.stop) if (start := r.start) else True

	@boolen
	def __len__(data, r, /):
		value = min(len(data), r.stop) - r.start
		return - (-value // r.step) if value > 0 else 0

	del boolen

	def __eq__(self, value, /):
		return (type(self) is type(value) and
			self.data is value.data and self.r == value.r)

	def _index(self, value, r, /) -> int:
		if r:
			data = self.data
			index = data.index
			start = r.start
			stop = r.stop
			return r.index(index(value) if not (start or stop)
				else index(value, start, stop))
		self.value_error(value)

	#peging
	def count(self, value, indices, /) -> int:
		if indices:
			try:
				return self.data.count(value, indices.start, indices.stop)
			except TypeError:
				return super().count(value)
		return 0

	def unpack(self, /) -> Iterator:
		return self.data[(r := self.r).start:r.stop:r.step]


class chain(SequenceView):
	'''Same as it.chain but as a sequence.'''
	__slots__ = ()
	data:TS

	__len__ = calcsize(sum)

	__bool__ = datamethod(any)

	def __init__(self, /, *sequences):
		super().__init__(*sequences)

	def __getitem__(self, index, /):
		data = self.data
		if type(index) is not slice:
			if index < 0:
				data = filter(None, reversed(data))
				for sequence in data:
					if (size := len(sequence)) >= -index:
						return sequence[index]
					index += size
			else:
				for sequence in filter(None, data):
					if index < (size := len(sequence)):
						return sequence[index]
					index -= size
			self.IndexError()

		size = [*get_sizes(data)]
		start, stop, step = index.indices(sum(size))
		key = step != 1
		values = []
		
		for seq, size in zip(data, size):
			values.append(seq := seq[start:stop:step])
			if key:
				start = (step - ((((size - 1) - start) % step)) - 1)
			
			elif start:
				if seq:
					start = 0
				else:
					start -= size
			
			if (stop := (stop - size)) <= 0:
				break
		
		return type(self)(*values)
			
	
	__iter__ = datamethod(from_iterable)

	def __reversed__(self, /):
		return from_iterable(MAP[2](reversed(self.data)))
	
	def __add__(self, value, /):
		if type(self) is type(value):
			self = self.__copy__()
			self.data += value.data
			return value
		return NotImplemented

	def cc_func(func, fmap, /):#Count and Contains Function decorator
		return lambda self, obj, / : func(fmap(self.data, irepeat(obj)))

	__contains__, count = map(cc_func, (any, sum), CC_MAP)

	del cc_func

	def index(self, value, start=0, stop:OPINT=None, /) -> int:
		data = self.data
		size = [*it.accumulate(get_sizes(data), initial=0)]
		if stop is None and not start:
			for data, size in zip(data, size):
				try:
					value = data.index(value)
				except ValueError as e:
					pass
				else:
					return (value + size)
		else:
			if start < 0:
				start += size[-1]
		
			if stop < 0:
				stop +=	 size[-1]
		
			for data, (size, n) in zip(data, it.pairwise(size)):
				if (r := (start - size)) < n and (n := (stop - size)) > 0:
					try:
						value = data.index(value, r, n)
					except ValueError as e:
						pass
					else:
						return value + size
		
		self.value_error(value)
	
	@classmethod
	def fromsequence(cls, data:Sequence[Sequence], /):
		(self := cls())._setattr('data', data)
		return self


@frozen_dataclass
class Repeat(Ranged):
	'''Same as it.repeat but as a sequence.'''
	value:Any

	def __init__(self, /, object:Any, times:int):
		super().__init__(range(times >= 0 and times))
		self._setattr('value', object)

	def __repr__(self, /):
		return f"{type(self).__name__}({self.value!r}, {self.r.stop!r})"
		
	def __len__(self, /):
		return self.r.stop
	
	def __iter__(self, /):
		return irepeat(self.value, self.r.stop)

	__reversed__ = __iter__
	def __contains__(self, obj, /):
		return True if self.r.stop and self.value == obj else False

	def __add__(self, value, /):
		if (cls := type(self)) is type(value):
			if (value := self.value) == value.value:
				return cls(value, self.r.stop + value.r.stop)
		return NotImplemented
	
	def _getitem(self, index:int, /):
		return self.value

	def _getslice(self, r, /):
		return type(self)(self.value, len(r))

	def count(self, value, /) -> int:
		return self.r.stop if value in self else 0

	def index(self, value, /) -> int:
		if value in self:
			return 0
		else:
			self.value_error(value)

	def tolist(self, /) -> list:
		return [self.value] * self.r.stop


RelativeSized = make_dataclass('RelativeSized', (('r', int),),
	frozen=True, slots=True, bases=(Size,))


class mul(RelativeSized):
	'''Emulates a data sequence multiplied r times.'''
	__slots__ = ()

	def __mul__(self, r, /):
		return self.__replace(r=self.r * r)

	def __add__(self, value, /):
		if type(self) is type(value):
			if (data := self.data) == data.data:
				return self._replace(r=self.r + data.r)
		return NotImplemented

	def __getitem__(self, index, /):
		try:
			floordiv, index = divmod(index, len(data := self.data))
			if div_index(floordiv // self.times):
				return data[index]
		except ZeroDivisionError:
			pass
		self.IndexError()

	def __len__(self, /):
		return len(self.data) * self.times

	def __contains__(self, value, /):
		return True if self.times and (value in self.data) else False

	def iterfunc(reverse, /):
		r = MAP[2] if reverse else None
		def __iter__(self, /):
			value = super().__iter__()
			if r:
				value = r(value)
			return from_iterable(value)
		return __iter__

	def count(self, value, /) -> int:
		return (r := self.times) and self.data.count(value) * r

	def index(self, value, start=0, stop:OPINT=None, /) -> int:
		index = self.data.index
		if (r := self.times):
			if stop is None and not start:
				return index(value)
			div, start = divmod(start, r)
			if div_index(div):
				return index(value, start, stop % r)
		self.value_error(value)

	def unpack(self, /) -> Sequence:
		return self.data * self.r


class Repeats(mul):
	'''Emulates a sequence with each elements repeated r times.'''
	__slots__ = ()

	__mul__ = SequenceView.__mul__
	
	def __getitem__(self, index, /):
		if (r := self.times):
			return self.data[index // r]
		else:
			self.IndexError()

	def iterfunc(reverse, /):
		def __iter__(self, /):
			data = self.data
			if reverse:
				data = reversed(data)
			return from_iterable(MAP[0](data, irepeat(self.r)))
		return __iter__

	def index(self, value, start=0, stop:OPINT=None, /) -> int:
		if not (r := self.times):
			self.value_error(value)
		
		index = self.data.index
		
		if stop is None and not start:
			return index(value) * r
		
		start, mod = divmod(start, r)
		return (index(value, start, stop//r) * r) + mod


class SubSequence(SequenceView):
	'''Base Class for sequences of sequences'''
	__slots__ = ()

	_index, _count = Sequence.index, Sequence.count

	_contains = Sequence.__contains__

	_check = checker(tuple)

	def __contains__(self, value, /):
		return self._check(value) and self._contains(value)

	def index(self, value, /, start:int=0, stop:OPINT=None) -> int:
		if self._check(value):
			return self._index(value, start, stop)
		else:
			self.value_error(value)

	def count(self, value, /) -> int:
		return self._count(value) if self._check(value) else 0


class Zip(SubSequence):
	"""Same as builtins.zip but as a sequence."""
	__slots__ = '_kwval'
	data:TS

	def __init__(self, /, *sequences:TS, strict:bool=False):
		super().__init__(sequence)
		self._setattr('_kwval', strict)

	__bool__ = zipbool(all)

	__len__ = calcsize(min)

	def __getitem__(self, index, /):
		data = self.data
		if (index_type := type(index)) is tuple:
			x,y = index
			return data[i][y]

		data = tuple(map(op.itemgetter(index), data))

		if index_type is slice:
			return type(self)(*data, strict=self._kwval)

		return data

	def __iter__(self, /):
		return zip(*self.data, strict=self._kwval)

	def __reversed__(self, revmap=MAP[2], /):
		return zip(*self._reversegen(self._levels(), self._kwval))

	@staticmethod
	def _reversegen(levels, r, /):
		for i, (data, level) in enumerate(levels):
			data = reversed(data)
			if level:
				if r:
					raise TypeError(f"Sequence #{i} has different size.")
				data = it.islice(data, level, None)
			yield data

	def _contains(self, value, /):
		try:
			self.index(value)
		except ValueError:
			return
		else:
			return True

	def _count(self, value, /):
		start = 0,
		try:
			for start in enumerate(self.iter_index(value)):
				pass
		except ValueError:
			return start[0]

	def _check(self, value, /):
		return type(value) is tuple and len(value) == len(self.data)

	def _levels(self, /) -> tuple[Sequence, ITII]:
		data = self.data
		n = irepeat(len(self))
		return zip(data, map(abs, map(op.sub, n, get_sizes(data))))

	def _index(self, values, start, stop, /):
		if data := self.data:
			if start is not None:
				indices = [seq.index(value, start,
					stop) for value, seq in zip(values, data)]
			
			else:
				indices = [*map(op.indexOf, data, values)]
			
			maxvalue = max(indices)
			stop = maxvalue + 1
			n = len(data)
			
			while indices.count(maxvalue) != n:
				iterable = enumerate(zip(data, values, indices))
			
				for index, (seq, value, start) in iterable:
					if start != maxvalue:
						indices[index] = seq.index(value, start + 1, stop)
			
			return maxvalue
		
		self.value_error(values)


class zip_longest(Zip):
	'''Same as it.zip_longest but as a sequence.'''
	__slots__ = ()

	def __init__(self, /, *sequences:TS, fillvalue=None):
		super().__init__(sequences, strict=fillvalue)

	__bool__ = zipbool(any)

	__len__ = calcsize(max)

	def __getitem__(self, index, /):
		if type(index) is slice:
			return super().__getitem__(index)
		default = self._kwval
		return tuple(get(data, index, default) if level else data[index]
			for data, level in self._levels())

	def __iter__(self, /):
		return it.zip_longest(*self.data, fillvalue=self._kwval)

	@staticmethod
	def _reversegen(levels, default, /):
		for data, level in levels:
			data = reversed(data)
			if level:
				data = it.chain(irepeat(default, start), data)
			yield data


class combinations(SubSequence, RelativeSized):
	'''Base Class for combinatoric sequences. A combinations subclass is a type
	of sequence that returns r-length sucessive tuples of different combinations
	of all elements in data.'''
	__slots__ = ()

	def __bool__(self, /):
		return not (r := self.r) or len(self.data) >= r


class nwise(combinations):
	'''Emulates tuples of every r elements of data.'''

	def __getitem__(self, index, /):
		return tuple(getitems(self.data, range(index, index + self.r)))

	def iterfunc(reverse, /):
		def __iter__(self, /):
			first = data = self.data

			if reverse:
				first = reversed(first)

			if not (r := self.r):
				return irepeat((), len(data))

			elif r < 3:
				return NWISE_ITER[r](first)

			else:
				r = range(1, r)
				first = iter(first)
				if not reverse:
					if hasattr(first, '__setstate__'):
						limit = it.islice
					else:
						limit = islice
						args = map(limit, irepeat(data), r, R_NONE)
					return zip(first, *args)
		return __iter__

	def __len__(self, /):
		return len(self.data) - (self.r - 1) if self else 0

	def __bool__(self, /):
		return self.r <= len(self.data)

	def _contains(func, /):
		def function(self, value, /):
			data = efficient_nwise(self.data, self.r)
			value = irepeat(deque(value))
			return func(map(op.eq, data, value))
		return function

	def _index(self, value, start, stop, /):
		return Sequence.sub(self.data, value, start, stop)

	_count = _contains(sum)

	@_contains
	def _contains(stream, /):
		return next(filter(None, stream), None)


class Product(combinations):
	'''Same as it.product but acts as a sequence.'''
	__slots__ = ()
	data:TS

	def __init__(self, /, *sequences, repeat=1):
		super().__init__(sequence, repeat)

	def __repr__(self, /):
		string = f"{type(self).__name__!r}{self.data!r}".removesuffix(')')
		return f'{string}, repeat={self.r!r})'

	__bool__ = datamethod(all)

	def __mul__(self, r, /):
		return type(self)(*self.data, repeat=self.r * r)

	def __getitem__(self, index, /):
		data, size, _, r, n = self.stats()
		index = range(n)[index]
		values = []

		for (data, size, r) in zip(data, size, r):
			values.append(data[((index % (size * r)) // r)])

		return tuple(values)


	def __len__(self, /):
		return math.prod(get_sizes(self.data)) ** self.r

	def iterfunc(r, /):
		def __iter__(self, /)  -> zip:
			data, size, count, times, _, = self.stats()
			first, *values = data
			del count[0]

			if not data:
				return iter(((),))

			if values:
				repeat, unchain, mapreversed = MAP[:3]
				data = repeat(values, count)

				if r:
					first = reversed(first)
					data = map(mapreversed, data)

				values[:] = unchain(data)
				values.insert(0, first)
				values[:-1] = unchain(map(repeat, values[:-1], repeat(times)))

				return zip(*values)

			return zip(first)
		
		return __iter__


	def _contains(func, fmap, /):
		return lambda self, value, /: func(fmap(self.data * self.r, value))

	def _check(self, value, /) -> bool:
		return type(value) is tuple and (
			len(value) // self.r) == len(self.data)

	r = list[int]

	def stats(self, /) -> tuple[tuple, r, r, r, int]:
		floordiv = MAP[-2]
		size = [*get_sizes(data := self.data)]
		size *= (repeat := self.r)
		*count, n = it.accumulate(size, op.mul, initial=1)
		times = [*floordiv(floordiv(irepeat(n), size), count)]
		return data * repeat, size, count, times, n

	del r

	def _index(self, value, start, stop, /):
		data, _, _, repeat, _, = self.stats()
		return math.sumprod(map(op.indexOf, data, value), repeat)

	_contains, _count = map(_contains, (all, math.prod), CC_MAP)


@frozen_dataclass
class enumerated(SubSequence):
	"""Same as builtins.enumerate but as a sequence."""
	start:int=0

	def __getitem__(self, index, /):
		data = self.data
		value = data[index]
		if index < 0:		
			index += len(data)
		return (index + self.start, value)

	def __iter__(self, /):
		return enumerate(self.data, self.start)
	
	def __reversed__(self, /):
		data = self.data	
		return zip(it.count(self.start + len(data) - 1, -1), reversed(data))

	def _check(self, value, /):
		return type(value) is tuple and len(value) == 2

	def _index(self, value, start, stop, /):
		data = self.data
		index, obj = value
		if start or (r := self.start):
			i = data.index(obj, start, stop)
			if (r + i) == index:
				return value
		else:
			try:
				if data[index] == value:
					return index
			except IndexError:
				pass
		self.value_error(value)

	def _count(self, value, /):
		return +self._contains(value)

	def _contains(self, value, /):
		index, obj = value
		if (r := self.start):
			index = abs(index - r)
		value = get(self.data, index, SENTINEL)
		return value is not SENTINEL and value == obj


@dataclass(frozen=True, slots=True)
class Progression(Ranged):
	'''Emulates an Arithmetic Progression:
	r = Arange indicating the indices of the progression.
	a1 = the first term of the progression.
	d = teh distance between each term.
	'''
	a1:Number=0
	d:Number=1

	def __contains__(self, number, /):
		return self._getindex(number) in self.r

	def rfunc(func, /):
		return lambda self, /:func(self.r)

	__len__, __bool__ = rfunc(len), rfunc(bool)

	del rfunc

	def _getslice(self, r, /):
		OSETATTR(new := type(self)(self.a1, self.d, n=0), 'r', r)
		return new

	def _getitem(self, index:int, /):
		return self.a1 + (index * self.d)

	def _getindex(self, number:Number, /):
		return (number - self.a1) / self.d

	@property
	def start(self, /) -> Number:
		return self._getitem(self.r.start)

	@property
	def step(self, /) -> Number:
		return self.d * self.r.step

	@property
	def stop(self, /) -> Number:
		return self._getitem(self.r.stop)

	@property
	def last(self, /) -> Number:
		return self._getitem(self.r[-1])


	def iterfunc(reverse, /):
		def __iter__(self, /):
			step = (r := self.r).step
			if reverse:
				start = self.last
				step = -step
			else:
				start = self.start
			return it.islice(it.count(start, self.step * step), len(r))
		return __iter__

	def count(self, number:Number, /) -> int:
		return self.r.count(self._getindex(number))

	def index(self, number:Number, /) -> int:
		return self.r.index(self._getindex(number))

	
	@classmethod
	@slicer
	def fromrange(cls, slicer, /):
		'''Create Progression from a range. The stop argument will not be
		preserved if (stop - last_range_number) != step'''
		if (step := slicer.step) is None:
			growing = step = 1
		
		else:
			check_step(step)
			growing = step > 0
		
		stop = slicer.stop

		if (start := slicer.start) is None:
			start = 1
			n = math.trunc(slicer.stop)
		
		elif (start == stop) or (growing and start > stop) or (start < stop):
			n = 0
		
		else:
			diff = stop - start if growing else start - stop
			n = math.ceil(diff / abs(step))
		
		return cls(range(n), start, step)


	@classmethod
	def create_with_size(cls, /, start:Number=0, step:Number=1, *, n):
		'''Returns a Progression form start with step of size n'''
		if n < 0:
			raise ValueError("The size of the progression must be >= 0")
		return cls(range(n), start, step)


del (maxsize, ITII, UserDict, bicompose, UserList, partializer, OPINT, TS,
	Iterator, Iterable, Generator, Callable, Sequence)