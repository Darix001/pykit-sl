from operator import concat
from itertools import repeat

__author__ = 'Dariel Buret'
__version__ = '1.0'
__date__ = '4/12/2025'


tools = ('cache', 'compose', 'dict', 'method', 'num', 
	'numpy', 'range', 'string')

__all__ = [*map(concat, tools, repeat('tools'))]

del concat, repeat