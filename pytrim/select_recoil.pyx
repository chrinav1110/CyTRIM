# select_recoil.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.math cimport sqrt, sin, cos, fabs
import numpy as np

cdef double MEAN_FREE_PATH = 1.0
cdef double PMAX = 1.0

def setup(double density):
    global MEAN_FREE_PATH, PMAX
    MEAN_FREE_PATH = density**(-1.0/3.0)
    PMAX = MEAN_FREE_PATH / sqrt(np.pi)

cpdef get_recoil_position(double[::1] pos, double[::1] dir, double[::1] dirp_out):
    cdef double free_path = MEAN_FREE_PATH

    # random p + phi
    cdef double p = PMAX * sqrt(np.random.rand())
    cdef double fi = 6.283185307179586 * np.random.rand()

    cdef double cos_fi = cos(fi)
    cdef double sin_fi = sin(fi)

    # find smallest component k
    cdef int k = 0
    if fabs(dir[1]) < fabs(dir[0]): k = 1
    if fabs(dir[2]) < fabs(dir[k]): k = 2

    cdef int i = (k + 1) % 3
    cdef int j = (i + 1) % 3

    cdef double cos_alpha = dir[k]
    cdef double sin_alpha = sqrt(dir[i]*dir[i] + dir[j]*dir[j])
    cdef double cos_phi = dir[i] / sin_alpha if sin_alpha != 0.0 else 1.0
    cdef double sin_phi = dir[j] / sin_alpha if sin_alpha != 0.0 else 0.0

    # direction of recoil
    cdef double di = cos_fi*cos_alpha*cos_phi - sin_fi*sin_phi
    cdef double dj = cos_fi*cos_alpha*sin_phi + sin_fi*cos_phi
    cdef double dk = - cos_fi*sin_alpha

    # put di/dj/dk in right indices
    dirp_out[i] = di
    dirp_out[j] = dj
    dirp_out[k] = dk

    # normalize
    cdef double n = sqrt(dirp_out[0]*dirp_out[0] + dirp_out[1]*dirp_out[1] + dirp_out[2]*dirp_out[2])
    if n != 0.0:
        dirp_out[0] /= n
        dirp_out[1] /= n
        dirp_out[2] /= n

    return free_path, p
