# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport sqrt, cos, sin, fabs
import numpy as np
cimport numpy as cnp

cdef double MEAN_FREE_PATH = 1.0
cdef double PMAX = 1.0

def setup(double density):
    global MEAN_FREE_PATH, PMAX
    MEAN_FREE_PATH = density**(-1.0/3.0)
    PMAX = MEAN_FREE_PATH / sqrt(np.pi)

cdef inline int _argmin_abs3(double a0, double a1, double a2) nogil:
    """First index of the minimum absolute value (NumPy argmin behavior)."""
    cdef double b0 = fabs(a0)
    cdef double b1 = fabs(a1)
    cdef double b2 = fabs(a2)
    cdef int k = 0
    cdef double best = b0
    if b1 < best:
        k = 1
        best = b1
    if b2 < best:
        k = 2
    return k

cpdef get_recoil_position(double[::1] pos, double[::1] dir):
    """
    Returns:
      free_path (float), p (float), dirp (np.ndarray[3]), pos_recoil (np.ndarray[3])
    """
    cdef double free_path = MEAN_FREE_PATH

    # collision point (C math)
    cdef double pc0 = pos[0] + free_path * dir[0]
    cdef double pc1 = pos[1] + free_path * dir[1]
    cdef double pc2 = pos[2] + free_path * dir[2]

    # randoms (NumPy under GIL)
    cdef double u1 = np.random.rand()
    cdef double u2 = np.random.rand()
    cdef double p  = PMAX * sqrt(u1)
    cdef double fi = 2.0 * 3.141592653589793 * u2

    cdef double cos_fi = cos(fi)
    cdef double sin_fi = sin(fi)

    # frame selection exactly like numpy argmin(abs(dir))
    cdef int k = _argmin_abs3(dir[0], dir[1], dir[2])
    cdef int i = (k + 1) % 3
    cdef int j = (i + 1) % 3

    cdef double cos_alpha = dir[k]
    cdef double sin_alpha
    if i == 0 and j == 1:
        sin_alpha = sqrt(dir[0]*dir[0] + dir[1]*dir[1])
    elif i == 0 and j == 2:
        sin_alpha = sqrt(dir[0]*dir[0] + dir[2]*dir[2])
    elif i == 1 and j == 2:
        sin_alpha = sqrt(dir[1]*dir[1] + dir[2]*dir[2])
    elif i == 1 and j == 0:
        sin_alpha = sqrt(dir[1]*dir[1] + dir[0]*dir[0])
    elif i == 2 and j == 0:
        sin_alpha = sqrt(dir[2]*dir[2] + dir[0]*dir[0])
    else:  # (2,1)
        sin_alpha = sqrt(dir[2]*dir[2] + dir[1]*dir[1])

    cdef double cos_phi = dir[i] / sin_alpha
    cdef double sin_phi = dir[j] / sin_alpha

    # recoil direction components
    cdef double d0 = 0.0
    cdef double d1 = 0.0
    cdef double d2 = 0.0

    cdef double comp_i = cos_fi*cos_alpha*cos_phi - sin_fi*sin_phi
    cdef double comp_j = cos_fi*cos_alpha*sin_phi + sin_fi*cos_phi
    cdef double comp_k = -cos_fi*sin_alpha

    if i == 0: d0 = comp_i
    elif i == 1: d1 = comp_i
    else: d2 = comp_i

    if j == 0: d0 = comp_j
    elif j == 1: d1 = comp_j
    else: d2 = comp_j

    if k == 0: d0 = comp_k
    elif k == 1: d1 = comp_k
    else: d2 = comp_k

    # normalize dirp (C math)
    cdef double n = sqrt(d0*d0 + d1*d1 + d2*d2)
    if n != 0.0:
        d0 /= n; d1 /= n; d2 /= n

    # recoil position
    cdef double pr0 = pc0 + p * d0
    cdef double pr1 = pc1 + p * d1
    cdef double pr2 = pc2 + p * d2

    # build NumPy outputs exactly like before
    cdef cnp.ndarray[cnp.float64_t, ndim=1] dirp = np.empty(3, dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] pos_recoil = np.empty(3, dtype=np.float64)
    dirp[0] = d0; dirp[1] = d1; dirp[2] = d2
    pos_recoil[0] = pr0; pos_recoil[1] = pr1; pos_recoil[2] = pr2

    return free_path, p, dirp, pos_recoil
