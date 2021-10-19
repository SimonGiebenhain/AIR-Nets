from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name = 'libvoxelize',
      ext_modules = cythonize("*.pyx"),
      include_dirs=[numpy.get_include()]
      )
