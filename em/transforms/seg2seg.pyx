cimport cython
cimport numpy as np
import ctypes
import numpy as np
import scipy.ndimage



cdef extern from 'cpp-seg2seg.h':
    long *CppMapLabels(long *segmentation, long *mapping, unsigned long nentries)
    long *CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries)
    long *CppForceConnectivity(long *segmentation, long zres, long yres, long xres)


# map the labels from this segmentation
def MapLabels(segmentation, mapping):
    # get the size of the data
    zres, yres, xres = segmentation.shape
    nentries = segmentation.size

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    cdef np.ndarray[long, ndim=1, mode='c'] cpp_mapping
    cpp_mapping = np.ascontiguousarray(mapping, dtype=ctypes.c_int64)

    cdef long *mapped_segmentation = CppMapLabels(&(cpp_segmentation[0,0,0]), &(cpp_mapping[0]), nentries)

    cdef long[:] tmp_segmentation = <long[:segmentation.size]> mapped_segmentation

    return np.reshape(np.asarray(tmp_segmentation), (zres, yres, xres))



# remove the components less than min size
def RemoveSmallConnectedComponents(segmentation, threshold=64):
    if threshold == 0: return segmentation

    nentries = segmentation.size
    zres, yres, xres = segmentation.shape

    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    
    # call the c++ function
    cdef long *updated_segmentation = CppRemoveSmallConnectedComponents(&(cpp_segmentation[0,0,0]), threshold, nentries)

    # turn into python numpy array
    cdef long[:] tmp_segmentation = <long[:segmentation.size]> updated_segmentation

    # reshape the array to the original shape
    thresholded_segmentation = np.reshape(np.asarray(tmp_segmentation), (zres, yres, xres))	
    return np.copy(thresholded_segmentation)



# reduce the labeling
def ReduceLabels(segmentation):
    # get the unique labels
    unique = np.unique(segmentation)

    # get the maximum label for the segment
    maximum_label = np.amax(segmentation) + 1

    # create an array from original segment id to reduced id
    mapping = np.zeros(maximum_label, dtype=np.int64) - 1
    # extraceullar maps to extracellular
    mapping[0] = 0

    # nothing else should map to zero
    index = 1
    for label in unique:
        # prevent extracellular material
        if label == 0: continue

        # set the mapping to this index and increment
        mapping[label] = index
        index += 1

    # return the forward and reverse mapping
    return mapping, unique



def ForceConnectivity(segmentation):
    # transform into c array
    cdef np.ndarray[long, ndim=3, mode='c'] cpp_segmentation
    cpp_segmentation = np.ascontiguousarray(segmentation, dtype=ctypes.c_int64)
    zres, yres, xres = segmentation.shape

    # call the c++ function
    cdef long *cpp_components = CppForceConnectivity(&(cpp_segmentation[0,0,0]), zres, yres, xres)

    # turn into python numpy array
    cdef long[:] tmp_components = <long[:zres*yres*xres]> cpp_components
    
    # reshape the array to the original shape
    components = np.reshape(np.asarray(tmp_components), (zres, yres, xres)).astype(np.int32)

    # find which segments have multiple components
    return components