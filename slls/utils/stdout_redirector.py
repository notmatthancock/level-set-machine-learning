"""
For hijacking stdout to the logger. Code from:

https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/

Example::

    from cStringIO import StringIO

    f = StringIO()

    with stdout_redirector(f):
        print "Things to be printed"
        print "Another thing", "that was printed"

    print("Hijacked from stdout: {}".format(f.getvalue()))
"""
import sys
from contextlib import contextmanager


@contextmanager
def stdout_redirector(stream):
    old_stdout = sys.stdout
    sys.stdout = stream

    try:
        yield
    finally:
        sys.stdout = old_stdout
