def setup_cython():
    from distutils.core import setup
    from distutils.extension import Extension
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
	
    import numpy
    ext_modules = []
    # malis pyx
    ext_modules += [Extension("em.lib.malis.malis_core", 
                             sources=["em/lib/malis/malis_core.pyx"], 
                             language='c++',extra_link_args=["-std=c++11"],
                             extra_compile_args=["-std=c++11", "-w"])]
    # warping pyx
    ext_modules += [Extension('em.lib.elektronn._warping',
                             sources=['em/lib/elektronn/_warping.pyx'],
                             extra_compile_args=['-std=c99', '-fno-strict-aliasing', '-O3', '-Wall', '-Wextra'])]

    # calculate vi
    ext_modules += [cythonize(
		    Extension(name='em.evaluation.comparestacks',
                    sources=['em/evaluation/comparestacks.pyx', 'em/evaluation/cpp-comparestacks.cpp'],
                    extra_compile_args=['-O4', '-std=c++0x'],
                    language='c++'))]
	
    # utils for vi calculation
    ext_modules += [cythonize(
		    Extension(name='em.transforms.distance',
        	    sources=['em/transforms/distance.pyx', 'em/transforms/cpp-distance.cpp'],
                    extra_compile_args=['-O4', '-std=c++0x'],
        	    language='c++'))]

    ext_modules += [cythonize(
	 	    Extension(name='em.transforms.seg2gold',
        	    sources=['em/transforms/seg2gold.pyx', 'em/transforms/cpp-seg2gold.cpp'],
                    extra_compile_args=['-O4', '-std=c++0x'],
                    language='c++'))]

    ext_modules += [cythonize(
		    Extension(name='em.transforms.seg2seg',
                    sources=['em/transforms/seg2seg.pyx', 'em/transforms/cpp-seg2seg.cpp'],
                    extra_compile_args=['-O4', '-std=c++0x'],
                    language='c++'))]
	
    setup(name='em_python',
       version='1.0',
       cmdclass = {'build_ext': build_ext}, 
       include_dirs=[numpy.get_include(),''], 
       packages=['em',
		 'em.evaluation',
		 'em.transforms',
                 'em.lib','em.data', 'em.model',
                 'em.prune','em.quant','em.util',
                 'em.lib.malis', 'em.lib.elektronn',
                 'em.lib/align_affine'],
       ext_modules = ext_modules)

if __name__=='__main__':
    # export CPATH=$CONDA_PREFIX/include:$CONDA_PREFIX/include/python2.7/ 
    # pip install --editable .
	setup_cython()
