cimport cython
cimport numpy as np
import numpy as np
import ctypes

cdef extern from 'cpp-seg2gold.h':
    long *CppMapping(long *segmentation, int *gold, long nentries, double low_threshold, double high_threshold)

def Mapping(segmentation, gold, low_threshold=0.10, high_threshold=0.80):
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[int, ndim=3, mode='c'] cpp_gold
    cpp_gold = np.ascontiguousarray(gold, dtype=ctypes.c_int32)

    max_segmentation = np.amax(segmentation) + 1

    cdef long *mapping = CppMapping(&(cpp_segmentation[0,0,0]), &(cpp_gold[0,0,0]), segmentation.size, low_threshold, high_threshold)

    cdef long[:] tmp_mapping = <long[:max_segmentation]> mapping;

    return np.asarray(tmp_mapping)