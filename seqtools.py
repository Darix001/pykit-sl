'''This library is directly inspired by itertools, as the idea of conceiving a library
for dealing with seqtools in a more efficient way.'''
from __future__ import annotations

import itertools as it, math, operator as op, collections.abc as abc, \
more_itertools as mit

from sys import maxsize
from numbers import Number
from typing import Any, NamedTuple
from collections.abc import Sequence
from dataclasses import dataclass, replace
from collections import deque, UserDict, UserList
from functools import wraps, update_wrapper as wrap

from .rangetools import rargs, slicer
from .methodtools import (
	fromcls,
	set_name,
	constfunc,
	add_method,
	MethodType,
	partializer,
	dunder_method,
	unassigned_method,
	)
from .composetools import simple_compose


from_iterable = it.chain.from_iterable
irepeat = it.repeat
div_index = {0, -1}.__contains__
data_method = dunder_method('data')
data_func = data_method.with_func

FROM_ITERTOOLS = 1
R_NONE = irepeat(None)
NEW_OBJ = classmethod(object.__new__)

mapper = partializer(map)

MAP = [it.repeat, from_iterable, reversed, it.islice,
op.itemgetter(slice(None, None, -1)), op.getitem,
op.floordiv, op.attrgetter('stop'), len, op.contains, op.countOf]

MAP[:] = map(mapper, MAP)

CC_MAP = MAP[-2:]
del MAP[-2:]
get_sizes = MAP.pop()

ITII = abc.Iterator[tuple[int, int]]
NWISE_ITER = {1:zip, 2:it.pairwise}
MAXSIZE_RANGE = range(maxsize)
SENTINEL = object()


def efficient_nwise(iterable:abc.Iterable, n:int) -> abc.Generator[deque]:
	'''Yields efficients nwise views of the given iterable re-using a
	collections.deque object.'''
	
	data = deque(it.islice(iterable := iter(iterable), n - 1), n)
	return it.dropwhile(data.append, iterable)


def getitems(data:Sequence, items, /) -> abc.Iterator:
	'''fetchs and Yields each item of the data object.'''
	return map(data.__getitem__, items)


def getslices(data:Sequence, subindices:abc.Iterator[abc.Iterator[int]]
	) -> abc.Iterator:
	'''fetchs and Yields each slices of the data object given a group of indices'''
	return map(data.__getitem__, it.starmap(slice, subindices))


def checker(cls, /):
	'''Creates a Check method for SubSequence subclasses'''

	return unassigned_method(
		lambda self, obj, /: obj.__class__ is cls and len(obj) == self.r)


def bool_method(func, /) -> abc.Callable:
	
	@wraps(func)
	def function(self, /):
		if self:
			return func(self)
		else:
			raise ValueError(
				f"attempt to get {func.__name__}"
				f"on empty {self.__class__.__name__}"
				)
	
	return function


def comb_len(cls, /) -> type:
	'''Decorator for combinations and permutation classes that computes their get_sizes based
	on their respective math function.'''

	func = getattr(math, cls.__name__[:4])
	
	def __len__(self, /):
		return func(len(self.data), self.r)
	add_method(cls, __len__)
	
	return cls


def multidata(cls, /) -> type:
	'''Decorator for sequence that accepts multiple sequence and base
	their size on the sizes of their sequences.'''
	namespace = vars(cls)
	if len_func := namespace.get('__len__'):
		def __len__(self, /):
			return len_func(get_sizes(self.data))
		add_method(cls, __len__)

	if bool_func := namespace.get('__bool__'):
		add_method(cls, data_func(bool_func), '__bool__')

	return cls


def efficient_slice(data:Sequence, slicer:slice):
	if isinstance(data, islice_):
		return data._replace(r=data.r[slicer])

	elif not slicer.start and slicer.stop is None:
		match slicer.step:
			case None|1:
				return SequenceView(data)

			case -1:
				return ReverseView(data)

	return efficient_slice(data, range(len(data))[slicer])
		

islice = slicer(efficient_slice)		



def repeat(object:Any, times:int) -> Repeat:
	return Repeat(range(times if times > 0 else 0), object)

def progression(start:Number, step:Number, /, n:int):
	if n < 0:
		raise ValueError("n must be a positive integer")
	return Count(range(n), start, step)


def get(data:Sequence, index:int, default:Any=None, /):
	'''Return the value for key if key is in the sequence, else default.'''
	try:
		return data[index]
	except IndexError:
		return default


def all_equals(data:Sequence, /) -> bool:
	'''Returns True if all items on sequence are equals'''
	return data.count(data[0]) == len(data)


def sub(data:Sequence, values:Sequence, start:int=0, stop:int=maxsize, /):
	first = values[0]
	values = islice_(values, MAXSIZE_RANGE[diff:])
	while diff:
		index = start = data.index(first, start, stop)
		for start, value in enumerate(values, start + diff):
			if diff := value != data[start]:
				start += diff
				break
	return index


class BaseSequence(Sequence):
	'''Base class for all classes in this module.'''
	__slots__ = ()
	iterfunc = None
	_replace = replace
	_new = NEW_OBJ
	
	def __init_subclass__(cls, /, iterfunc=None):
		if factory := cls.iterfunc:
			cls.iterfunc = None
			iterfunc = factory(None)
			add_method(cls, factory(True), '__reversed__')

		elif iterfunc:
			if iterfunc == FROM_ITERTOOLS:
				func = getattr(it, cls.__name__)
			else:
				func = iterfunc
			def __iter__(self, /):
				return func(**vars(self))
			iterfunc = __iter__

		else:
			return

		add_method(cls, iterfunc)

	
	def value_error(self, value, /):
		raise ValueError(f"{value!r} not in {self.__class__.__name__}")

	def index_error(self, /):
		raise IndexError(f"{self.__class__.__name__} index out of range.")

		
@dataclass(frozen=True)
class SequenceView(BaseSequence):
	'''A view over a new sequence.'''
	__slots__ = 'data'
	data:Sequence
	index = count = fromcls(UserList)

	def __getitem__(self, index, /):
		data = self.data
		if type(index) is slice:
			if len(r := range(n := len(data))[index]) == n:
				if r.step > 0:
					return self
				else:
					return ReverseView(data)
			return islice_(data, r)
		else:
			return data[index]

	__reversed__ = __bool__ = data_method

	__len__ = __contains__ = __iter__ = fromcls(UserDict)

	def __mul__(self, n, /):
		return mul(self.data, n)


class ReverseView(SequenceView):
	'''Reversed View of a sequence data'''
	__slots__ = ()
	__reversed__ = SequenceView.__iter__
	__iter__ = SequenceView.__reversed__

	def __getitem__(self, index, /):
		data = self.data
		if type(index) is slice:
			if len(r := range(n := len(data) -1, -1, -1)[index]) == n:
				if step < 0:
					return SequenceView(data)
				else:
					return self
			return islice_(data, r)
		else:
			return data[~index]

	def index(self, value:Any, start=0, stop=None, /):
		n = len(data := self.data)
		getindex = data.index
		if not start and stop is None:
			return ~getindex(value)
		else:
			return ~getindex(value, ~stop + n, ~start + n)


class Size(SequenceView):
	__slots__ = ()
	'''Base Class for sequence wrappers that transform their sequence size.'''

	def __bool__(self, /):
		return True if self.data and self.r else False


@dataclass(frozen=True)
class ranged(BaseSequence):
	r:range

	def __getitem__(self, index, /):
		r = self.r[index]
		return self._getslice(r) if type(index) is slice else self._getitem(r)

	def __contains__(self, value, /):
		if indices := self.r:
			return self._contains(value, indices)
		else:
			return False

	def index(self, value, start=0, stop=None, /):
		if indices := self.r[start:stop]:
			return self._index(value, indices)
		else:
			self.value_error(value)

	def count(self, value, start=0, stop=None, /):
		if indices := self.r[start:stop]:
			return self._count(value, indices)
		else:
			return 0


class indexed(Size, ranged):
	r:Sequence[int]

	def __len__(self):
		return len(self.r) if self.data else 0
	
	def iterfunc(reverse, /):
		def __iter__(self, /):
			indices = self.r
			return getitems(data, reversed(indices) if reverse else indices)
		return __iter__

	def _getitem(self, index:int, /):
		return self.data[index]

	def _getslice(self, r:range, /):
		return self._replace(r=r)

	def _index(self, obj, indices, /):
		return indices.index(self.data.index(obj))

	def _count(self, obj, indices, /):
		return sum(map(indices.count, mit.locate(self.data, obj)))

	def unpack(self, /):
		return getitems(self.data, self.r)


class islice_(indexed):
	def __reversed__(self, /):
		size = len(data := self.data)
		if start := (r := self.r).start:
			return getitems(data, r)
		return it.islice(it.chain((None,), reversed(data)),
			size - start, size - self.stop, abs(self.step))
	
	def __iter__(self, /):
		gen = iter(data := self.data)
		start, stop, step = rargs(r := self.r)
		if start:
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
		return (self.__class__ is value.__class__ and
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

	def unpack(self, /):
		return self.data[(r := self.r).start:r.stop:r.step]


@multidata
class chain(BaseSequence):
	'''Same as it.chain but as a sequence.'''
	__slots__ = 'data'
	data:tuple[Sequence]

	__len__ = sum

	__bool__ = any

	def __init__(self, /, *sequences):
		self.data = sequences

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
		self = self._replace(data = values)
		
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
		return self
			
	__iter__ = data_func(from_iterable)

	__reversed__ = data_func(simple_compose(from_iterable, MAP[2], reversed))
	
	def __add__(self, value, /):
		if self.__class__ is value.__class__:
			self = self.__copy__()
			self.data += value.data
			return value
		return NotImplemented

	def cc_func(func, fmap, /):#Count and Contains Function decorator
		return lambda self, obj, / : func(fmap(self.data, irepeat(obj)))

	__contains__, count = map(cc_func, (any, sum), CC_MAP)

	def index(self, value, start=0, stop=None, /) -> int:
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
	def from_iterable(cls, data:Sequence[Sequence], /):
		(self := cls._new()).data = data
		return self
	

@dataclass(init=False, frozen=True)
class RelativeSized(Size):
	r:int


class Sized(ranged):
	__slots__ = ()
	times = property(op.attrgetter('r.stop'))

	def __len__(self, /):
		return self.r.stop


@dataclass(frozen=True)
class Repeat(Sized):
	'''Same as it.repeat but as a sequence.'''
	value:Any

	def __add__(self, value, /):
		if (cls := type(self)) is type(value):
			if (value := self.value) == value.value:
				return cls(value, self.times + value.times)
		return NotImplemented
	
	def __iter__(self, /):
		return irepeat(self.value, self.times)

	__reversed__ = __iter__
		
	def __contains__(self, obj, /):
		return True if self.times and self.value == obj else False

	def __repr__(self, /):
		return f"repeat({self.value!r}, {self.times!r})"

	def _getitem(self, index:int, /):
		return self.value

	def _getslice(self, r:range, /):
		return self._replace(r=range(len(r)))

	def count(self, value, /) -> int:
		return self.times if value in self else 0

	def index(self, value, /) -> int:
		if value in self:
			return 0
		else:
			self.value_error(value)

	def tolist(self, /) -> list:
		return [self.value] * self.times


@dataclass(frozen=True)
class mul(RelativeSized):
	'''Emulates a data sequence multiplied r times.'''

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

	def iterfunc(reverse, /):
		r = MAP[2] if reverse else None
		def __iter__(self, /):
			value = super().__iter__()
			if r:
				value = r(value)
			return from_iterable(value)
		return __iter__

	def __len__(self, /):
		return len(self.data) * self.times

	def __contains__(self, value, /):
		return True if self.times and (value in self.data) else False

	def count(self, value, /) -> int:
		return (r := self.times) and self.data.count(value) * r

	def index(self, value, start=0, stop=None, /) -> int:
		index = self.data.index
		if (r := self.times):
			if stop is None and not start:
				return index(value)
			div, start = divmod(start, r)
			if div_index(div):
				return index(value, start, stop % r)
		self.value_error(value)

	def unpack(self, /):
		return self.data * self.r


class Repeats(mul):
	'''Emulates a sequence with each elements repeated r times.'''

	def __getitem__(self, index, /):
		if (r := self.times):
			return self.data[index // r]
		self.IndexError()

	def iterfunc(reverse, fmap=MAP[0], /):
		def __iter__(self, /):
			data = self.data
			if reverse:
				data = reversed(data)
			return from_iterable(fmap(data, irepeat(self.times)))
		return __iter__

	def index(self, value, start=0, stop=None, /) -> int:
		if not (r := self.times):
			self.value_error(value)
		index = self.data.index
		if stop is None and not start:
			return index(value) * r
		start, mod = divmod(start, r)
		return (index(value, start, stop//r) * r) + mod

	def unpack(self, /):
		cls = type(data := self.data)
		return cls(MAP[0](data, irepeat(self.r)))


class SubSequence(BaseSequence):
	'''Base Class for sequences of sequences'''

	_index, _count = Sequence.index, Sequence.count

	_contains = Sequence.__contains__

	_check = checker(tuple)

	def __contains__(self, value, /):
		return self._check(value) and self._contains(value)

	def index(self, value, /, start=0, stop=maxsize):
		if self._check(value):
			if start is SENTINEL:
				start = None
			return self._index(value, start, stop)
		else:
			self.value_error(value)

	def count(self, value, /):
		return self._count(value) if self._check(value) else 0


@dataclass(frozen=True)
class chunked(RelativeSized, SubSequence):
	'''split the given sequence in iterables of n size.'''

	def __post_init__(self, /):
		if op.index(self.r) < 0:
			raise ValueError("n must be greater than zero")

	def __getitem__(self, index, /):
		data = self.data
		n = self.r
		if (index_type := type(index)) is tuple:
			row, col = index
			return data[(row * n) + col]
		elif index_type is slice:
			...
		else:
			index *= n
			if data := self._getitem(data, slice(index, (index + n) or None)):
				return data
		self.IndexError()

	def __setitem__(self, index, value, /):
		index *= (n := self.r)
		slice_obj = slice(index, (index + n) or None)
		try:
			self.data[index]
		except Exception as e:
			raise e
	
	def __len__(self, /):
		return -(-len(self.data)//self.r)

	def iterfunc(r, /):
		def __iter__(self, /):
			return map(MethodType(islice_, self.data),
				it.starmap(range, self._indexes(r)))
		return __iter__

	# _getitem = islice_.fromslice

	_check = checker(islice)

	def _indexes(self, reverse, rev=MAP[2], /) -> ITII:
		indices = range(0, len(self.data) + (n := self.r), n)
		if reverse:
			indices = reversed(indices)
		indices = it.pairwise(indices)
		if reverse:
			indices = fmap(indices)
		return indices

	def subiter(self, /):
		return it.starmap(self.data.__getitem__, self._indexes(None))

	flatten = Sequence.__iter__


class matrix(chunked):
	'''Acts as it like the given sequence was splitted in rows of r size.'''

	def iterfunc(reverse, /):
		def __iter__(self, /):
			return getslices(self.data, self._indexes(reverse))
		return __iter__

	def unpack(self, /):
		return type(self.data)(self)

	def _check(self, value, /) -> bool:
		return value.__class__ is self.data.__class__ and len(value) == self.r

	_getitem  = op.getitem


class batched(chunked, iterfunc=FROM_ITERTOOLS):
	"""Same as it.batched but as a sequence."""

	def __reversed__(self, /):
		return MAP[4](it.batched(reversed(self.data), self.r))

	def _getitem(self, slice_obj, /):
		return tuple(getitems(self.data, MAXSIZE_RANGE[slice_obj]))


@multidata
class Zip(SubSequence):
	"""Same as builtins.zip but as a sequence."""
	data:tuple[Sequence]

	def __init__(self, /, *sequences:tuple[Sequence], strict:bool=False):
		self.data = sequences
		self.strict = strict

	__bool__ = all

	__len__ = min

	def __getitem__(self, index, /):
		data = self.data
		if (index_type := type(index)) is tuple:
			x,y = index
			return data[i][y]
		else:
			data = tuple(map(op.itemgetter(index), data))
		return self._replace(data=data) if index_type is slice else data

	def __setitem__(self, index, value):
		data = self.data
		if (index_type := type(index)) is tuple:
			x,y = index
			data[x][y] = value
		else:
			for seq, obj in zip(data, value, strict=True):
				seq[index] = obj

	def __iter__(self, /):
		return zip(*self.data, strict=self.strict)

	def __reversed__(self, revmap=MAP[2], /):
		return zip(*self._reversegen(self._levels(), self.strict))

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
		return value.__class__ is tuple and len(value) == len(self.data)

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


@multidata
class zip_longest(Zip):
	'''Same as it.zip_longest but as a sequence.'''

	def __init__(self, /, *sequences, fillvalue=None):
		self.data = sequences
		self.fillvalue = fillvalue

	__bool__ = any

	__len__ = max

	def __getitem__(self, index, /):
		if type(index) is slice:
			return super().__getitem__(index)
		default = self.fillvalue
		return tuple(get(data, index, default) if level else data[index]
			for data, level in self._levels())

	def __iter__(self, /):
		return it.zip_longest(*self.data, fillvalue=self.fillvalue)

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


def product(*args, repeat:int=1):
	return product(args, repeat)


@multidata
@dataclass(frozen=True)
class Product(combinations, RelativeSized):
	'''Same as it.product but acts as a sequence.'''
	data:tuple[Sequence]

	__bool__ = all

	def __mul__(self, r, /):
		new = cls._new()
		new.data = self.data
		new.r = self.r * r

	def __getitem__(self, index, /):
		data, size, _, r, n = self.stats()
		if ~n <= index < n:
			values, index = [], range(n)[index]
			for (data, size, r) in zip(data, size, r):
				values.append(data[((index % (size * r)) // r)])
			return tuple(values)
		else:
			self.IndexError()

	def __len__(self, /):
		return math.prod(get_sizes(self.data)) ** self.r

	__mul__ = mul.__mul__

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
		return value.__class__ is tuple and (len(value)
			// self.r) == len(self.data)

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




@dataclass(frozen=True)
class enumerated(SubSequence, iterfunc=enumerate):
	"""Same as builtins.enumerate but as a sequence."""
	start:int=0

	def __getitem__(self, index, /):
		data = self.data
		value = data[index]
		if index < 0:		
			index += len(data)
		return (index + self.start, value)
	
	def __reversed__(self, /):
		data = self.data	
		return zip(it.count(self.start + len(data) - 1, -1), reversed(data))

	__len__, __bool__ = SequenceView.__len__, SequenceView.__bool__

	def _check(self, value, /):
		return value.__class__ is tuple and len(value) == 2

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


class cycle(SequenceView):
	'''Same as it.cycle but emulating a sequence.'''
	__slots__ = ()
	
	def __getitem__(self, index, /):
		data = self.data
		return data[index % len(data)]

	__iter__ = data_func(simple_compose(from_iterable, irepeat))

	__reversed__ = data_func(simple_compose(from_iterable, MAP[2], irepeat))

	def __len__(self, /):
		return math.inf

	__bool__ = constfunc(True)

	def count(self, value, /) -> int | float:
		return math.inf if value in self.data else 0


@dataclass(frozen=True)
class Count(Sized):	
	'''The sequence-like version of it.count'''	
	start:Number=0
	step:Number=1

	def __contains__(self, number, /):
		index, mod = self._getindex(number)
		return index >= 0 and not mod

	def __repr__(self, /):
		return f"progression({self.start!r}, {self.step!r}, n={self.times!r})"

	def _getitem(self, index:int, /):
		return self.start + (index * self.step)

	def _getslice(self, r:range, ):
		step = self.step
		start = self.start
		if nstart := r.start:
			start += step * nstart
		return type(self)(range(len(r)), start, step * r.step)

	def _getindex(self, number:Number, /):
		return divmod(number - self.start, self.step)

	@property
	def stop(self, /):
		return self._getitem(self.times)

	def iterfunc(r, /):
		def __iter__(self, /):
			step = self.step
			times = self.times
			if r:
				start = self._getitem(times - 1)
				step = -step
			else:
				start = self.start
			return it.islice(it.count(start, step), times)
		return __iter__

	def count(self, number:Number, /) -> int:
		return +(number in self)

	def index(self, number:Number, /):
		index, mod = self._getindex(number)
		if index < 0 or mod:
			self.value_error(number)
		return math.trunc(index)


del (maxsize, abc, ITII, UserDict, simple_compose, UserList, set_name,
	partializer, dunder_method, unassigned_method, fromcls)