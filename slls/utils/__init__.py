"""
A variety of helper functions.

The C scripts each have a corresponding "f2py signature file" (.pyf). 
We use NumPy's `f2py` utility to create Python wrapper libraries for 
these utilities. `f2py` is intended to be used to wrap Fortran, but wraps 
C also, provided we use the line `intent(C)` in the signature files.

I chose C over Fortran because NumPy in Python uses row-major ordering, so
any Fortran functions would have to copy all the data into the routines. 
This adds non-trivial computation time for large volumes. Note that `f2py`
utility doesn't handle boolean valued arrays (this is a known
bug), and copies the data irrespective of row/column-major ordering.

-Matt Hancock, 2018
"""
