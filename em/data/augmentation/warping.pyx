"""
Adapted from ELEKTRONN (http://elektronn.org/).
"""

import numpy as np


cdef extern from 'cpp/warping/c-warping.h':
    int fastwarp2d_opt(const float * src,
               float * dest_d,
               const int sh[3],
               const int ps[3],
               const float rot,
               const float shear,
               const float scale[2],
               const float stretch_in[2])
    int fastwarp3d_opt_zxy(const float * src,
                     float * dest_d,
                     const int sh[4],
                     const int ps[4],
                     const float rot,
                     const float shear,
                     const float scale[3],
                     const float stretch_in[4],
                     const float twist_in)


def warp2dFast(img, patch_size, rot=0, shear=0, scale=(1,1), stretch=(0,0)):
    """
    Create warped mapping for a spatial 2D input image.
    The transformation is done w.r.t to the *center* of the image.

    Parameters
    ----------

    img: array
      The array must be 3-dimensional (ch,x,y) and larger/equal the patch size
    patch_size: 2-tuple
      Patch size *excluding* channel: (px, py).
      The warping result of the input image is cropped to this size
    rot: float
      Rotation angle in deg for rotation around z-axis
    shear: float
      Shear angle in deg for shear w.r.t xy-diagonal
    scale: 2-tuple of float
      Scale per axis
    stretch: 2-tuple of float
      Fraction of perspective stretching from the center (where stretching is always 1)
      to the outer border of image per axis. The 4 entry correspond to:

      - X stretching depending on Y
      - Y stretching depending on X


    Returns
    -------

    img: np.ndarray
      Warped image (cropped to patch_size)
    """
    assert len(img.shape)==3
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180

    scale   = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale   = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    img = np.ascontiguousarray(img, dtype=np.float32)
    cdef float [:, :, :] img_view = img
    cdef float * in_ptr = &img_view[0, 0, 0]

    cdef int [:] in_sh_view = np.ascontiguousarray(img.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    out_arr = np.zeros((img.shape[0],)+tuple(patch_size), dtype=np.float32)
    cdef float [:, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0]

    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr  = &ps_view[0]

    fastwarp2d_opt(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear, scale_ptr, stretch_ptr)
    return out_arr


def _warp2dFastLab(lab, patch_size, img_sh, rot, shear, scale, stretch):
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180

    scale   = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale   = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    new_lab = np.zeros((1,)+img_sh, dtype=np.float32)
    off = list(map(lambda x: (x[0]-x[1])//2, zip(img_sh, lab.shape)))
    new_lab[0, off[0]:lab.shape[0]+off[0], off[1]:lab.shape[1]+off[1]] = lab
    lab = new_lab
    cdef float [:, :, :] lab_view = lab
    cdef float * in_ptr = &lab_view[0, 0, 0]

    cdef int [:] in_sh_view = np.ascontiguousarray(lab.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    out_shape = list(map(lambda x: x[0]-2*x[1], zip(patch_size, off)))
    out_shape = (1,) + tuple(out_shape)

    out_arr = np.zeros(out_shape, dtype=np.float32)
    cdef float [:, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0]

    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr  = &ps_view[0]

    fastwarp2d_opt(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear, scale_ptr, stretch_ptr)
    out_arr = out_arr.astype(np.int16)[0]
    return out_arr


def warp3dFast(img, patch_size, rot=0, shear=0, scale=(1,1,1), stretch=(0,0,0,0), twist=0):
    """
    Create warped mapping for a spatial 3D input image.
    The transformation is done w.r.t to the *center* of the image.

    Note that some transformations are not applied to the z-axis. This makes this function simpler
    and it is also better for anisotropic data as the different scales are not mixed up then.

    Parameters
    ----------

    img: array
      The array must be 4-dimensional (z,ch,x,y) and larger/equal the patch size
    patch_size: 3-tuple
      Patch size *excluding* channel: (pz, px, py).
      The warping result of the input image is cropped to this size
    rot: float
      Rotation angle in deg for rotation around z-axis
    shear: float
      Shear angle in deg for shear w.r.t xy-diagonal
    scale: 3-tuple of float
      Scale per axis
    stretch: 4-tuple of float
      Fraction of perspective stretching from the center (where stretching is always 1)
      to the outer border of image per axis. The 4 entry correspond to:

      - X stretching depending on Y
      - Y stretching depending on X
      - X stretching depending on Z
      - Y stretching depending on Z

    twist: float
      Dependence of the rotation angle on z in deg from center to outer border

    Returns
    -------

    img: np.ndarray
      Warped array (cropped to patch_size)

    """
    assert len(img.shape)==4

    # Rotation, shear, twist.
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180
    twist = twist * np.pi / 180

    # Scale.
    scale = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    # Perspective stretch.
    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    # Image.
    img = np.ascontiguousarray(img, dtype=np.float32)
    cdef float [:, :, :, :] img_view = img
    cdef float * in_ptr = &img_view[0, 0, 0, 0]

    # Image shape.
    cdef int [:] in_sh_view = np.ascontiguousarray(img.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    # Output.
    out_shape = (patch_size[0], img.shape[1], patch_size[1], patch_size[2])
    out_arr = np.zeros(out_shape, dtype=np.float32)
    cdef float [:, :, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0, 0]

    # Output shape.
    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr = &ps_view[0]

    fastwarp3d_opt_zxy(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear,
                        scale_ptr, stretch_ptr, twist)
    return out_arr


def _warp3dFastLab(lab, patch_size, img_sh, rot, shear, scale, stretch, twist):
    n_chann = lab.shape[1]
    lab_sh  = (lab.shape[0], lab.shape[2], lab.shape[3])

    # Rotation, shear, twist.
    rot   = rot   * np.pi / 180
    shear = shear * np.pi / 180
    twist = twist * np.pi / 180

    # Scale.
    scale = np.array(scale, dtype=np.float32, order='C', ndmin=1)
    scale = 1.0/scale
    cdef float [:] scale_view = scale
    cdef float * scale_ptr = &scale_view[0]

    # Perspective stretch.
    stretch = np.array(stretch, dtype=np.float32, order='C', ndmin=1)
    cdef float [:] stretch_view = stretch
    cdef float * stretch_ptr = &stretch_view[0]

    # Label.
    new_lab_sh = (img_sh[0], n_chann, img_sh[1],img_sh[2])
    new_lab = np.zeros(new_lab_sh, dtype=np.float32)
    off = list(map(lambda x: (x[0]-x[1])//2, zip(img_sh, lab_sh)))
    new_lab[off[0]:lab_sh[0]+off[0], :, off[1]:lab_sh[1]+off[1], off[2]:lab_sh[2]+off[2]] = lab
    lab = new_lab
    cdef float [:, :, :, :] lab_view = lab
    cdef float * in_ptr = &lab_view[0, 0, 0, 0]

    # Label shape.
    cdef int [:] in_sh_view = np.ascontiguousarray(lab.shape, dtype=np.int32)
    cdef int * in_sh_ptr = &in_sh_view[0]

    out_shape = patch_size
    out_shape = (out_shape[0], n_chann, out_shape[1], out_shape[2])
    out_arr = np.zeros(out_shape, dtype=np.float32)
    cdef float [:, :, :, :] out_view = out_arr
    cdef float * out_ptr = &out_view[0, 0, 0, 0]

    # Output shape.
    cdef int [:] ps_view = np.ascontiguousarray(out_arr.shape, dtype=np.int32)
    cdef int * ps_ptr = &ps_view[0]

    fastwarp3d_opt_zxy(in_ptr, out_ptr, in_sh_ptr, ps_ptr, rot, shear,
                        scale_ptr, stretch_ptr, twist)
    # out_arr = out_arr.astype(np.int16)[:,0]
    return out_arr

def warp2dJoint(img, lab, patch_size, rot, shear, scale, stretch):
    """
    Warp image and label data jointly. Non-image labels are ignored i.e. lab must be 3d to be warped

    Parameters
    ----------

    img: array
      Image data
      The array must be 3-dimensional (ch,x,y) and larger/equal the patch size
    lab: array
      Label data (with offsets subtracted)
    patch_size: 2-tuple
      Patch size *excluding* channel for the image: (px, py).
      The warping result of the input image is cropped to this size
    rot: float
      Rotation angle in deg for rotation around z-axis
    shear: float
      Shear angle in deg for shear w.r.t xy-diagonal
    scale: 3-tuple of float
      Scale per axis
    stretch: 4-tuple of float
      Fraction of perspective stretching from the center (where stretching is always 1)
      to the outer border of image per axis. The 4 entry correspond to:

      - X stretching depending on Y
      - Y stretching depending on X

    Returns
    -------

    img, lab: np.ndarrays
      Warped image and labels (cropped to patch_size)

    """
    if len(lab.shape) == 2:
        lab = _warp2dFastLab(lab, patch_size, img.shape[1:], rot, shear, scale, stretch)

    img = warp2dFast(img, patch_size, rot, shear, scale, stretch)
    return img, lab


def warp3dJoint(img, lab, patch_size, rot=0, shear=0, scale=(1, 1, 1), stretch=(0, 0, 0, 0), twist=0):
    """
    Warp image and label data jointly. Non-image labels are ignored i.e. lab must be 3d to be warped

    Parameters
    ----------

    img: array
      Image data
      The array must be 4-dimensional (z,ch,x,y) and larger/equal the patch size
    lab: array
      Label data (with offsets subtracted)
    patch_size: 3-tuple
      Patch size *excluding* channel for the image: (pz, px, py).
      The warping result of the input image is cropped to this size
    rot: float
      Rotation angle in deg for rotation around z-axis
    shear: float
      Shear angle in deg for shear w.r.t xy-diagonal
    scale: 3-tuple of float
      Scale per axis
    stretch: 4-tuple of float
      Fraction of perspective stretching from the center (where stretching is always 1)
      to the outer border of image per axis. The 4 entry correspond to:

      - X stretching depending on Y
      - Y stretching depending on X
      - X stretching depending on Z
      - Y stretching depending on Z

    twist: float
      Dependence of the rotation angle on z in deg from center to outer border

    Returns
    -------

    img, lab: np.ndarrays
      Warped image and labels (cropped to patch_size)

    """
    if len(lab.shape) == 3:
        lab = _warp3dFastLab(lab, patch_size, np.array(img.shape)[[0, 2, 3]], rot, shear, scale, stretch, twist)

    img = warp3dFast(img, patch_size, rot, shear, scale, stretch, twist)
    return img, lab


def warp3d(img, patch_size, rot=0, shear=0, scale=(1, 1, 1), stretch=(0, 0, 0, 0), twist=0):
    return warp3dFast(img, patch_size, rot, shear, scale, stretch, twist)

def warp3dLab(lab, patch_size, size, rot=0, shear=0, scale=(1, 1, 1), stretch=(0, 0, 0, 0), twist=0):
    return _warp3dFastLab(lab, patch_size, size, rot, shear, scale, stretch, twist)

### Utilities #################################################################
###############################################################################


def getCornerIx(sh):
    """Returns array-indices of corner elements for n-dim shape"""

    def getGrayCode(n, n_dim):
        if n == 0:
            return np.zeros(n_dim, dtype=np.int)
        return np.array([(n // 2**i) % 2 for i in range(max(n_dim, int(np.ceil(np.log2(n)))))])

    sh = np.array(sh) - 1  ###TODO
    n_dim = len(sh)
    ix = []
    for i in xrange(2**n_dim):
        ix.append(getGrayCode(i, n_dim))

    ix = np.array(ix)
    corners = ix * sh
    return corners


def _warpCorners2d(sh, corners, rot=0, shear=0, scale=(1, 1), stretch=(0, 0), plot=False):
    """
    Create warped coordinates of corners
    """
    rot = rot * np.pi / 180
    shear = shear * np.pi / 180
    scale = np.array(scale)
    scale = 1.0 / scale
    stretch = np.array(stretch)
    corners = corners.astype(np.float).copy()

    x_center_off = float(sh[0]) / 2 - 0.5
    y_center_off = float(sh[1]) / 2 - 0.5

    stretch[0] /= x_center_off
    stretch[1] /= y_center_off

    x = corners[:, 0] - x_center_off
    y = corners[:, 1] - y_center_off

    xt = x * (scale[0] + stretch[0] * y)
    yt = y * (scale[1] + stretch[1] * x)
    u = xt * np.cos(rot - shear) - yt * np.sin(rot + shear) + x_center_off
    v = yt * np.cos(rot + shear) + xt * np.sin(rot - shear) + y_center_off

    return np.array([u, v]).T


def _warpCorners3d(sh, corners, rot=0, shear=0, scale=(1, 1, 1), stretch=(0, 0, 0, 0), twist=0):
    """
    Create warped coordinates of corners
    """
    rot = rot * np.pi / 180
    shear = shear * np.pi / 180
    twist = twist * np.pi / 180
    scale = np.array(scale)
    scale = 1.0 / scale
    stretch = np.array(stretch)
    corners = corners.astype(np.float).copy()

    z_center_off = float(sh[0]) / 2 - 0.5
    x_center_off = float(sh[1]) / 2 - 0.5
    y_center_off = float(sh[2]) / 2 - 0.5

    stretch[0] /= x_center_off
    stretch[1] /= y_center_off
    stretch[2] /= z_center_off
    stretch[3] /= z_center_off
    twist /= z_center_off

    z = corners[:, 0] - z_center_off
    x = corners[:, 1] - x_center_off
    y = corners[:, 2] - y_center_off

    w = z * scale[2] + z_center_off
    rot = rot + (z * twist)

    xt = x * (scale[0] + stretch[0] * y + stretch[2] * z)
    yt = y * (scale[1] + stretch[1] * x + stretch[3] * z)
    u = xt * np.cos(rot - shear) - yt * np.sin(rot + shear) + x_center_off
    v = yt * np.cos(rot + shear) + xt * np.sin(rot - shear) + y_center_off

    return np.array((w, u, v)).T


def getRequiredPatchSize(patch_size, rot, shear, scale, stretch, twist=None):
    """
    Given desired patch size and warping parameters:
    return required size for warping input patch
    """
    patch_size = np.array(patch_size)
    corners = getCornerIx(patch_size)

    if len(patch_size) == 2:
        coords = _warpCorners2d(patch_size, corners, rot, shear, scale, stretch)
    elif len(patch_size) == 3:
        coords = _warpCorners3d(patch_size, corners, rot, shear, scale, stretch, twist)

    eff_size = np.ceil(coords.max(axis=0) - coords.min(axis=0))  # effective range
    left_exc = np.floor(np.abs(np.minimum(coords.min(axis=0), 0)))  # how much image needs to be added left
    right_exc = np.ceil(np.maximum(coords.max(axis=0) - patch_size + 1, 0))
    total_exc = np.maximum(left_exc, right_exc)  # how much image must be added centrally
    req_size = patch_size + 2 * total_exc

    return req_size.astype(np.int), eff_size.astype(np.int), left_exc.astype(np.int)


def getWarpParams(patch_size, amount=1.0, **kwargs):
    """
    To be called from CNNData. Get warping parameters + required warping input patch size.
    """
    if amount > 1:
        print 'WARNING: warpAugment amount > 1 this requires more than 1.4 bigger patches before warping'
    rot_max = 15 * amount
    shear_max = 3 * amount
    scale_max = 1.1 * amount
    stretch_max = 0.1 * amount
    n_dim = len(patch_size)

    # Data-specific max.
    if 'scale_max' in kwargs:
       scale_max = kwargs['scale_max']

    shear = shear_max * 2 * (np.random.rand() - 0.5)
    if n_dim == 3:
        twist = rot_max * 2 * (np.random.rand() - 0.5)
        rot = min(rot_max - abs(twist), rot_max * (np.random.rand()))
        scale = 1 - (scale_max - 1) * np.random.rand()
        scale = (scale, scale, 1)
        stretch = stretch_max * 2 * (np.random.rand(4) - 0.5)
    elif n_dim == 2:
        rot = rot_max * 2 * (np.random.rand() - 0.5)
        scale = 1 - (scale_max - 1) * np.random.rand(2)
        stretch = stretch_max * 2 * (np.random.rand(2) - 0.5)
        twist = None

    # DEBUG
    if 'rot' in kwargs:  # 0
       rot = kwargs['rot']
    if 'shear' in kwargs:  # 0
       shear = kwargs['shear']
    if 'scale' in kwargs:  # (1,1,1)
       scale = kwargs['scale']
    if 'stretch' in kwargs:  # (1,1,1,1)
       stretch = kwargs['stretch']
    if 'twist' in kwargs:  # 0
       twist = kwargs['twist']

    req_size, _, _ = getRequiredPatchSize(patch_size, rot, shear, scale,
                                          stretch, twist)

    return req_size, rot, shear, scale, stretch, twist



