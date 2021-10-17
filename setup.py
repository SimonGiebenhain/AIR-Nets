try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions

mcubes_module = Extension(
    'models.utils.libmcubes.mcubes',
    sources=[
        'models/utils/libmcubes/mcubes.pyx',
        'models/utils/libmcubes/pywrapper.cpp',
        'models/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'models.utils.libmise.mise',
    sources=[
        'models/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'models.utils.libsimplify.simplify_mesh',
    sources=[
        'models/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)





# Gather all extension modules
ext_modules = [
    mcubes_module,
    mise_module,
    simplify_mesh_module
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
