# This is basically swiped from:
# https://github.com/pypa/python-manylinux-demo/

mkdir -p wheelhouse

# Run the `build-wheels.sh` script in the manylinux docker container
sudo docker run --rm -it -e PLAT=manylinux1_x86_64 -v \
    `pwd`:/io quay.io/pypa/manylinux1_x86_64 \
    bash /io/build-wheels.sh
