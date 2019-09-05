import os
from setuptools import setup
from setuptools.command.build_ext import build_ext
import subprocess


PKG_NAME = 'lsml'

VERSION_MAJOR = 0
VERSION_MINOR = 0
VERSION_MICRO = 1
VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_MICRO}"


class CompileCSource(build_ext):
    """ Builds c utilities
    """
    def run(self):

        # Get the C compiler from env, default=gcc
        cc = os.environ.get('CC', 'gcc')

        # Build the c utils
        cmd = f'make CC={cc} -C {os.path.join(PKG_NAME, "util", "_cutil")}'
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

        build_ext.run(self)


def write_version():
    with open(os.path.join(PKG_NAME, '_version.py'), 'w') as f:
        f.write(f'version = "{VERSION}"')


if __name__ == '__main__':

    write_version()

    setup(
        author='Matt Hancock',
        author_email='not.matt.hancock@gmail.com',
        cmdclass={
            'build_ext': CompileCSource
        },
        description='Level set machine learning for image segmentation',
        include_package_data=True,
        install_requires=[
            'h5py',
            'matplotlib',
            'numpy',
            'scikit_fmm',
            'scikit_image',
            'scikit_learn',
            'scipy',
        ],
        license='MIT',
        name=PKG_NAME,
        packages=[PKG_NAME],
        url='https://github.com/notmatthancock/level-set-machine-learning/',
        version=VERSION,
    )
