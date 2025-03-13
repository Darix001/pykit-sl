'''Seq'''
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


from .methodtools import (
	fromcls,
	set_name,
	add_method,
	MethodType,
	partializer,
	dunder_method,
	unassigned_method,
	)
from .composetools import simple_compose


from_iterable = it.chain.from_iterable
irepeat = it.repeat
rargs = op.attrgetter('start', 'stop', 'step')
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
ROOT12 = math.sqrt(12)


def efficient_nwise(iterable:abc.Iterable, n:int) -> abc.Generator[deque]:
	'''Yields efficients nwise views of the given iterable re-using a
	collections.deque object.'''
	
	data = deque(it.islice(iterable := iter(iterable), n - 1), n)
	for _ in map(data.append, iterable):
		yield data


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


def counter_expand_method(func, /) -> abc.Callable:
	@wraps(func)
	def function(self, n:int, /):
		return func(self, self.step * n)
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

def slicer(func, /):
	return wrap(lambda obj, /, *args: func(obj, slice(*args)), func)


@slicer
def islice(data:Sequence, /, slicer):
	if isinstance(data, islice_):
		return data._replace(r=data.r[slicer])

	elif slicer.start is slicer.stop is None:
		match slicer.step:
			case None|1:
				return SequenceView(data)

			case -1:
				return ReverseView(data)

	return islice_(data, MAXSIZE_RANGE[slicer])

		


def repeat(object:Any, times:int) -> Repeat:
	return Repeat(range(times if times > 0 else 0), object)


def full(fill_value:Number, size:int) -> Full:
	return Full(range(times if times > 0 else 0), fill_value)



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
	data:Sequence

	__reversed__ = __bool__ = data_method

	__len__ = __contains__ = __iter__ = fromcls(UserDict)

	__getitem__ = __mul__ = index = count = fromcls(UserList)


class ReverseView(SequenceView):
	'''Reversed View of a sequence data'''
	def __getitem__(self, index, /):
		data = self.data
		if type(index) is slice:
			data = data[slice(*rargs(range(len(data))[index][::-1]))]
			return type(self)(data)
		else:
			return data[~index]

	__reversed__ = SequenceView.__iter__

	__iter__ = SequenceView.__reversed__

	def index(self, value:Any, start=0, stop=None, /):
		n = len(data := self.data)
		getindex = data.index
		if not start and stop is None:
			return ~getindex(value)
		else:
			return ~getindex(value, ~stop + n, ~start + n)


class Size(SequenceView):
	'''Base Class for sequence wrappers that transform their sequence size.'''

	def __bool__(self, /):
		return True if self.data and self.r else False


@dataclass(frozen=True)
class ranged(BaseSequence):
	r:range

	def __getitem__(self, index, /):
		r = self.r[index]
		if type(index) is slice:
			return self._replace(r=r)
		else:
			return self._getitem(self, r)

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
	
	def iterfunc(reverse, /):
		def __iter__(self, /):
			indices = self.r
			return getitems(data, reversed(indices) if reverse else indices)
		return __iter__

	def _getitem(self, index, /):
		return self.data[index]

	def _index(self, obj, indices, /):
		data = self.data

	def _count(self, obj, indices, /):
		data = self.data

	def __len__(self):
		return len(self.r) if self.data else 0

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

	def __ne__(self, value, /):
		return (self.__class__ is not value.__class__ or
			 self.data is not value.data or self.r != value.r)

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

	@classmethod
	def fromslice(cls, data:Sequence, slice_obj:slice, /):
		return cls(data, MAXSIZE_RANGE[slice_obj])


@multidata
class chain(BaseSequence):
	'''Same as it.chain but as a sequence.'''
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
	

@dataclass(frozen=True)
class Repeat(ranged):
	'''Same as it.repeat but as a sequence.'''
	value:Any

	def __mul__(self, times, /):
		return _replace(self, indices=range(self.times * times))

	def __add__(self, value, /):
		if (cls := type(self)) is type(value):
			if (value := self.value) == value.value:
				return cls(value, self.times + value.times)
		return NotImplemented
	
	def __iter__(self, /):
		return irepeat(self.value, self.times)

	__reversed__ = __iter__

	def _replace(self, indices, /):
		return _replace(self, indices=range(len(indices)))
	
	def __contains__(self, obj, /):
		return True if self.times and self.value == obj else False

	def _getitem(self, index, /):
		return self.value

	def count(self, value, /) -> int:
		return self.times if value in self else 0

	def index(self, value, /) -> int:
		if value in self:
			return 0
		else:
			self.value_error(value)

	def tolist(self, /):
		return [self.value] * self.times

	times = property(op.attrgetter('indices.stop'))
	
	def __len__(self, /):
		return self.r.stop

@dataclass(init=False, frozen=True)
class Sized(Size):
	r:int


@dataclass(frozen=True)
class mul(Sized):
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


class repeats(mul):
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

	def index(self, value, /, start=0, stop=maxsize):
		if self._check(value):
			if start is SENTINEL:
				start = None
			return self._index(value, start, stop)
		else:
			self.value_error(value)

	def count(self, value, /):
		return self._count(value) if self._check(value) else 0

	def __contains__(self, value, /):
		return self._check(value) and self._contains(value)

	_check = checker(tuple)


@dataclass(frozen=True)
class chunked(Sized, SubSequence):
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

	_getitem = islice_.fromslice

	_check = checker(islice)

	def _indexes(self, reverse, rev=MAP[2], /) -> ITII:
		indices = MAXSIZE_RANGE[:len(self.data) + (n := self.r):n]
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


class combinations(SubSequence, Sized):
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
class Product(combinations, Sized):
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


class InfiniteSequence(SequenceView):
	'''Base class for infinite sequence.'''
	def __len__(self, /):
		return math.inf

	__bool__ = irepeat(True).__next__

	def __getitem__(self, index, /):
		is_slice = type(index) is slice
		if not is_slice and (index := op.index(index)) < 0:
			raise TypeError("Negative index is not supported.")
		return self._getitem(index, is_slice)


class cycle(InfiniteSequence):
	'''Same as it.cycle but emulating a sequence.'''

	__iter__ = data_func(simple_compose(from_iterable, irepeat))

	__reversed__ = data_func(simple_compose(from_iterable, MAP[2], irepeat))

	def _getitem(self, index:int, is_slice, /):
		data = self.data
		return data[index % len(data)]

	def count(self, value, /) -> int | float:
		return math.inf if value in self.data else 0


@dataclass(order=True)
class BaseCounter:
	'''Case class for counter-like classes.'''
	start:Number=0
	step:Number=1

	def _intersect_update(self, other, /):
		'''The algorithm to get the first intersection point was provided
		by chatGPT
		#By assigning self.start as the final start variable we save few lines
		of code.
		start = self.start 
		b = other.start
		s1 = self.step
		s2 = other.step
		diff = Difference between starts
		g = greatest common divissor between steps'''
		start, b = self.start, other.start

		if (ss := (step := self.step) == (s2 := other.step)) == 1:
			if start < b:
				start = b
		
		else:
			s1 = step
		
			if ss:
				g = step
			else:
				if s1 < (g := s2):
					step, g = g, step

				if step % g:
					step *= g
					g = 1

			# If g doesn't divide diff, no solution
			if (diff := b - start) % g:
				return

			# Solve for the smallest non-negative n
			# Modular inverse of (s1 // g) mod (s2 // g)
			mod_inv = (1 / (s1 // g)) % (s2g := s2 // g)
			
			# First intersection point
			start += ((diff // g * mod_inv) % s2g) * s1 

		self.step = step
		self.start = start
		return True


	def _index_to_value(self, index, /):
		return self.start + (index * self.step)

	def _value_to_index(self, number, /):
		return divmod(number - self.start, self.step)
		

class count(BaseCounter, InfiniteSequence, iterfunc=FROM_ITERTOOLS):
	'''The sequence-like version of it.count'''	
	stop:float=math.inf

	def _getitem(self, index, is_slice, /):
		transform = self._index_to_value
		if is_slice:
			if not (start := index.start):
				start = self.start
			else:
				start = transform(start)

			if (step := index.step) is not None:
				step *= self.step

			elif step < 0:
				raise TypeError("Step value can't be negative.")

			else:
				step = self.step

			if (stop := index.stop) is None:
				return type(self)(start, step)
			
			elif not step:
				return repeat(start, stop - start)
			
			else:
				return Arange(start, transform(stop), step)

		else:
			return transform(index)


	def __contains__(self, number, /):
		index, mod = self._value_to_index(number)
		return index >= 0 and not mod

	def __and__(self, obj, /):
		if isinstance(obj, COUNTABLE):
			self = self.copy()
			if obj and (args := self._intersect_update(obj)):
				if (stop := obj.stop) is not self.stop:
					self.__class__ = Arange
					self.stop = stop
				return self
			else:
				return Arange(zero := self.start * 0, zero)
			
		else:
			return NotImplemented


	def __iand__(self, obj, /):
		if (cls := type(self)) is type(obj):
			return self if self._intersect_update(obj) else Arange(0, 0)
		else:
			return NotImplemented

	def __add__(self, obj, /):
		type(self)(self.start + obj, self.step)

	def __radd__(self, obj, /):
		type(self)(obj + self.start, self.step)

	def __sub__(self, obj, /):
		type(self)(self.start + obj, self.step)

	def __rsub__(self, obj, /):
		type(self)(obj + self.start, self.step)

	def count(self, number, /) -> int:
		return number in self[start:stop]

	def index(self, number, /):
		index, mod = self._value_to_index(number)
		if index < 0 or mod:
			self.value_error(number)
		return math.trunc(index)
	
	@counter_expand_method
	def expandleft(self, steps, /):
		self.start -= steps

	def popleft(self, /) -> Number:
		self.start = (start := self.start) + self.step
		return start

	def copy(self, /):
		return type(self)(self.start, self.step)


@dataclass(frozen=True)
class enumerated(SubSequence, InfiniteSequence, iterfunc=enumerate):
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


@dataclass(init=False)
class Arange(BaseCounter, BaseSequence):
	'''A mutable numeric range, with math, statistics and aritmethical
	methods, as well as support for set operations like intersection or the
	& operator.
	Example:
	number_list = Arange(0, 10, 2.)
	number_list[-3]	
	#prints 4.0

	 '''
	stop:Number=0

	@slicer
	def __init__(self, slicer, /):
		self.start = 0 if (start := slicer.start) is None else start
		self.stop = slicer.stop
		
		if (step := slicer.step) is None:
			self.step = 1
		
		elif not step:
			raise ValueError("Step argument must not be zero.")
		
		else:
			self.step = step

	
	def __repr__(self, /):
		string = f"{self.__class__.__name__}({self.start!r}, {self.stop!r}"
		if (step := self.step) == 1:
			return string + ')'
		else:
			return f"{string}, {step})"

	def __getitem__(self, index, /):
		get_value = self._index_to_value
		step = self.step
		if type(index) is slice:
			start, stop, istep = index.indices(len(self))
			return type(self)(get_value(start), get_value(stop), step * istep)
		else:
			stop = self.stop
			if (index := op.index(index)) < 0:
				index *= step
				index = math.ceil((stop + index - self.start) / step)
				if index >= 0:
					return get_value(index)

			elif (value := get_value(index)) < stop:
				return value

			raise IndexError("Arange Index out of range")

	def __len__(self, /):
		size = math.ceil((self.stop - self.start) / self.step)
		return size if size > 0 else 0

	def __bool__(self, /):
		if self.step > 0:
			return self.stop > self.start
		else:
			return self.start > self.stop

	def iter_method(func, /):
		@wraps(func)
		def function(self, /):
			return it.islice(func(it.count, self), len(self))
		return function

	@iter_method
	def __iter__(count, self, /):
		return count(self.start, self.step)

	@iter_method
	def __reversed__(count, self, /):
		return count(self._last, -self.step)

	del iter_method

	def __array__(self, /):
		from numpy import arange
		return arange(self.start, self.stop, self.step)	

	def __contains__(self, value, /):
		index, mod = self._value_to_index(value)
		return not mod and index >= 0 and value < self.stop

	def __invert__(self, /):
		return type(self)(~self.start, ~self.stop, -self.step)

	def __pos__(self, /):
		return type(self)(+self.start, +self.stop, +self.step)

	def __neg__(self, /):
		return type(self)(-self.start, -self.stop, -self.step)

	def __add__(self, x, /):
		return self._newlims(self.start + x, self.stop + x)

	def __radd__(self, x, /):
		return self._newlims(x + self.start, x + self.stop)

	def __iadd__(self, x, /):
		self.start += x
		self.stop += x
		return self

	def __sub__(self, x, /):
		return self._newlims(self.start - x, self.stop - x)

	def __rsub__(self, x, /):
		return self._newlims(x - self.start, x - self.stop)

	def __isub__(self, x, /):
		self.start -= x
		self.stop -= x
		return self

	def __mul__(self, x, /):
		return type(self)(self.start * x, self.stop * x, self.step * x)

	def __rmul__(self, x, /):
		return type(self)(x * self.start, x * self.stop, x * self.step)

	def __imul__(self, x, /):
		self.start *= x
		self.stop *= x
		self.step *= x
		return self

	def __truediv__(self, x, /):
		return type(self)(self.start / x, self.stop / x, self.step / x)

	def __rtruediv__(self, x, /):
		return type(self)(x / self.start, x / self.stop, x / self.step)

	def __itruediv__(self, x, /):
		self.start /= x
		self.stop /= x
		self.step /= x
		return self

	def __floordiv__(self, x, /):
		return type(self)(self.start // x, self.stop // x, self.step // x)

	def __rfloordiv__(self, x, /):
		return type(self)(x // self.start, x // self.stop, x // self.step)

	def __ifloordiv__(self, x, /):
		self.start //= x
		self.stop //= x
		self.step //= x
		return self


	def __and__(self, obj, /):
		if isinstance(obj, COUNTABLE):
			(self := self.copy())._intersect_update(obj)
			return self
		else:
			return NotImplemented

	
	def __iand__(self, obj, /):
		if isinstance(obj, COUNTABLE):
			self._intersect_update(obj)
			return self
		else:
			return NotImplemented


	def _newlims(self, start, stop, /):
		return type(self)(start, stop, self.step)


	def clear(self, /):
		if self:
			if self.step > 0:
				self.stop = self.start
			else:
				self.start = self.stop


	# def order_func(func, /):
	# 	@wraps(func)
	# 	def function(self, obj, /):
	# 		if type(self) is not type(obj):
	# 			return NotImplemented
			
	# 		elif len(self) == len(obj):
	# 			return func(self.start, obj.start, self.step, obj.step)

	# 		else:
	# 			raise TypeError("Can't compare Aranges with different sizes.")
	# 	return function

	
	def __eq__(self, obj, /):
		#sbv = same boolean value
		#nv = no values
		if isinstance(obj, Arange):
			nv  = (sbv := bool(self) == bool(ob)) == False
			return (sbv and nv or self.start == obj.start and
				self.step == obj.step and self._last == other._last)

		else:
			return NotImplemented


	@property
	def shape(self, /):
		return len(self),

	@property
	def ndim(self, /):
		return 1

	flat, _args, pyrange = map(property,
		(iter, rargs, op.methodcaller('tocls', range)))

	@property
	def _last(self, /) -> Number:
		self._index_to_value(self._lastindex)

	@property
	def _empty(self, /):
		return type(self)(zero := self.start * 0, zero)

	def _intersect_update(self, obj, /):
		if not self:
			return

		elif obj and (has_intersection := super()._intersect_update(obj)):
			if self.stop > (stop := obj.stop):
				self.stop = stop
			return has_intersection

		else:
			self.clear()


	@set_name
	def imag(name, /):
		attrmap = mapper(op.attrgetter(name))
		return property(lambda self, /: type(self)(*attrmap(rargs(self))))

	denominator = numerator = real = imag


	def count(self, value, /) -> int:
		return +(value in self)

	def index(self, value, /) -> int:
		index, mod = self._value_to_index(value)
		if not mod and index >= 0 and value < self.stop:
			return math.trunc(index)
		else:
			self.value_error(value)


	@counter_expand_method
	def expand(self, steps, /) -> Number:
		self.stop += steps

	@counter_expand_method
	def walk(self, steps, /) -> Number:
		self += steps

	
	def pop_method(func, /):
		@wraps(func)
		def function(self, /):
			if self:
				return func(self, self.step)
			else:
				raise IndexError("Pop from empty range")
		return function

	@pop_method
	def popleft(self, step, /) -> Number:
		self.start = (start := self.start) + step
		return start

	@pop_method
	def pop(self, step, /) -> Number:
		value = self._last
		self.stop -= step
		return value

	del pop_method


	def reverse(self, /):
		step = -self.step
		start = self.start
		self.start = last = self._last
		self.stop = start + step
		self.step = step

	def copy(self, /):
		return type(self)(*self._args)

	def sum(self, /) -> Number:
		start = self.start
		if self:
			return (start + self._last) * len(self) / 2
		else:
			return start * 0

	def st_method(func, /):
		@wraps(func)
		def function(self, /):
			if self:
				return func(self.step, (((n := len(self)) **2) - 1) / n)
			else:
				return math.nan
		return function

	@st_method
	def std(d, n, /):
		return (d / ROOT12) * math.sqrt(n)

	@st_method
	def var(d, n, /):
		return (d**2 / 12) * n

	del st_method

	
	@bool_method
	def mean(self, /) -> Number:
		return (self.start + self._last) / 2


	def minmax(func, /):
		@bool_method
		@wraps(func)
		def function(self, /) -> Number:
			return func(self, self.step > 0)
		return function

	@minmax
	def min(self, growing, /) -> Number:
		return self.start if growing else self._last

	@minmax
	def max(self, growing, /) -> Number:
		return self._last if growing else self.start
	
	@property
	def _lastindex(self, /) -> int:
		return len(self) - 1

	@minmax
	def argmax(self, growing, /):
		return self._lastindex if growing else 0

	@minmax
	def argmin(self, growing, /):
		return 0 if growing else self._lastindex


	def median_method(func, /):
		@bool_method
		@wraps(func)
		def function(self, /):
			return self._index_to_value(func(len(self)))
		return function

	@median_method
	def median_low(n, /) -> Number:
		i, mod = divmod(n, 2)
		return i if mod else i - 1

	@median_method
	def median_high(n, /) -> Number:
		return n // 2

	del median_method


	def tocls(self, cls, /) -> range:
		return cls(*self._args)

	@classmethod
	def fromobj(cls, obj, /):
		return cls(*rargs(obj))


	def adjust(self, /):
		'''Adjust the stop argument of the range to match exactly with the step.
		If the arange is empty, the arange is best left unchanged.
		
		Example:
		x = Arange(0, 95, 4)
		x.adjust()
		print(x) #prints Arange(0, 96, 4)
		'''
		last, step = self._last, self.step
		if self and self.stop - last != self.step:
			self.stop = last + step

	
	def intersection_update(self, /, *ranges):
		'''Update self by the intersection between multiple ranges.'''
		#If there is any empty range, there is no intersection.
		if self:
			if all(ranges) and all(map(super()._intersect_update, ranges)):
				if self.stop > (stop := min(MAP[-1](ranges))):
					self.stop = stop
			else:
				self.clear()

	def intersection(self, /, *ranges):
		'''Returns the intersection between multiple ranges as a new Arange'''
		(self := self.copy()).intersection_update(*ranges)
		return self
		

COUNTABLE = BaseCounter | range

del (maxsize, abc, ITII, UserDict, simple_compose, UserList, set_name,
	partializer, dunder_method, unassigned_method, fromcls)