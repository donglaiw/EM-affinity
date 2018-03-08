from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name='distance',
        include_dirs=[np.get_include()],
        sources=['distance.pyx', 'cpp-distance.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    ),
    Extension(
        name='seg2gold',
        include_dirs=[np.get_include()],
        sources=['seg2gold.pyx', 'cpp-seg2gold.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    ),
    Extension(
        name='seg2seg',
        include_dirs=[np.get_include()],
        sources=['seg2seg.pyx', 'cpp-seg2seg.cpp'],
        extra_compile_args=['-O4', '-std=c++0x'],
        language='c++'
    )
]

setup(
    name='transforms',
    ext_modules = cythonize(extensions)
)