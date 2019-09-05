### C utilities

These modules are imported into Python modules 
and wrapped via `ctypes`. For example, the `_masked_gradient.c`
C file is compiled to a shared `*.so` where it is then imported 
and used by the `lsml.gradient.masked_gradient` Python module.