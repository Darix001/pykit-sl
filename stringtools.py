from array import array

class array(array):
    '''Array subclass allows to call the from unicode method using the lshift
    (<<) operator. Example:
    data = array('u')
    data << "Hello World!"
    print(data.tounicode())
    >> Hello World!
    '''
    __slots__ = ()
    __lshift__ = write = array.fromunicode


def char_increment(x:int = 1, /) -> Callable[(str,), str]:
    '''A callable that increments string chars x steps'''
    x = range(x, x+1114112)
    return lambda self,/: self.translate(x)