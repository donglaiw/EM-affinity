from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name='comparestacks',
        include_dirs=[np.get_include()],
        sources=['comparestacks.pyx', 'cpp-comparestacks.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    ),
]

setup(
    name='comparestacks',
    ext_modules = cythonize(extensions)
)
