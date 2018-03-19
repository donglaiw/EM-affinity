from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc
from Cython.Distutils import build_ext
import numpy as np

def getExt_model():
    return []


def getExt_data():
    return [Extension('em_pth.data.augmentation.warping',
                 sources=['em_pth/data/augmentation/warping.pyx'],
                 extra_compile_args=['-std=c99', '-fno-strict-aliasing', '-O3', '-Wall', '-Wextra'])]

def setup_cython():

    ext_modules = []
    ext_modules += getExt_model()
    ext_modules += getExt_data()

    setup(name='em_pytorch',
       version='1.0',
       cmdclass = {'build_ext': build_ext}, 
       include_dirs=[np.get_include(), get_python_inc()], 
       packages=['em',
                 'em.optim','em.data', 'em.model',
                 'em.app','em.util'
                 ],
       ext_modules = ext_modules)
if __name__=='__main__':
    # export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/include/python2.7/ 
    # pip install --editable .
	setup_cython()
