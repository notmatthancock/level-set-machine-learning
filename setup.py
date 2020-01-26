import os
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
import subprocess


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
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: C',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Image Recognition',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Visualization',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
        ],
        description='Level set machine learning for image segmentation',
        ext_modules=[
            Extension(
                name=f'{PKG_NAME}.util.masked_gradient',
                sources=[
                    os.path.join(
                        PKG_NAME, 'util', '_cutil', 'masked_gradient.c'
                    )
                ],
                extra_compile_args=['-std=c99', '-DMI_CHECK_INDEX=0']
            ),
        ],
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
        packages=find_packages(),
        python_requires='>=3.6',
        url='https://github.com/notmatthancock/level-set-machine-learning/',
        version=VERSION,
    )
