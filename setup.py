import os
from setuptools import setup


PKG_NAME = 'lsml'

VERSION_MAJOR = 0
VERSION_MINOR = 0
VERSION_MICRO = 1
VERSION = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_MICRO}"


def write_version():
    with open(os.path.join(PKG_NAME, '_version.py'), 'w') as f:
        f.write(f'version = "{VERSION}"')


if __name__ == '__main__':

    write_version()

    setup(
        author='Matt Hancock',
        author_email='not.matt.hancock@gmail.com',
        description='Level set machine learning for image segmentation',
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
