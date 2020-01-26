#!/bin/bash

# This script is a modified version of that from:
# https://github.com/pypa/python-manylinux-demo/

set -e -x

# Compile wheels
for ver in cp36-cp36m cp37-cp37m cp38-cp38
do
    echo Building $ver
    pip=/opt/python/$ver/bin/pip
    $pip install numpy
    $pip install /io/
    $pip wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/lsml*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done
