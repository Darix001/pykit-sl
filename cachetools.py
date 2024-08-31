'''The following module describes efficients and simple tools for caching.
All this tools are usefull only for functions that only holds'''

from collections import defaultdict
from itertools import islice, count
from functools import update_wrapper


class Cache(defaultdict):
	'''Defaultdict that pass missing key to default_factory as an argument.'''
	__slots__ = ()
	__call__ = dict.__getitem__

	def __missing__(self, key, /):
		self[key] = key = self.default_factory(key)
		return key


def _cache_func(load, /):

	def _cache_func(func = None, /, value = None, maxsize = None):
		if func:
			data, iterable = load(func, value, maxsize)
			
			def cache(x:int, /):
				if (n := len(data)) <= x:
					data.extend(islice(iterable, (x + 1) - n))
				return data[x]
	
			return cache

		return lambda func,/: _cache_func(func, value, maxsize)

	return update_wrapper(_cache_func, load)


def _getcounter(start, size, /) -> range | count:
	'''Returns a itertools.count object if size is None else a range object.'''
	return range(start, size) if size is not None else count(start)


@_cache_func
def cumcache(func, value, size, /):
	'''Accumulated numeric cache'''
	return (value := [value]), map(func, value, _getcounter(1, size))


@_cache_func
def enumcache(func, data, size, /):
	'''Numeric Cache'''
	if data is None:
		data, start = [], 0
	else:
		start = len(data)
	return data, map(func, _getcounter(start, size))


del defaultdict