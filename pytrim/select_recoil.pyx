# select_recoil.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

"""Create the recoil position for the next collision.

Currently, only amorphous targets are supported. The free path length to
the next collision is assumed to be constant and equal to the atomic
density to the power -1/3.

Available functions:
    setup: setup module variables.
    get_recoil_position: get the recoil position.
"""

from libc.math cimport sqrt, sin, cos, fabs, M_PI
import numpy as np
from libc.stdlib cimport rand, RAND_MAX

cdef double MEAN_FREE_PATH = 1.0
cdef double PMAX = 1.0

cdef inline double urand() nogil:
    # uniform [0,1)
    return rand() / <double>RAND_MAX


cdef unsigned long long rng_state = 88172645463393265

cdef inline double fast_rand() nogil:
    global rng_state
    cdef unsigned long long x = rng_state
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    rng_state = x
    return (x * 2685821657736338717ULL) / 18446744073709551616.0


def setup(double density):
    global MEAN_FREE_PATH, PMAX
    MEAN_FREE_PATH = density**(-1.0/3.0)
    PMAX = MEAN_FREE_PATH / sqrt(M_PI)




cdef void get_recoil_position_c(double[::1] pos, double[::1] dir, double[::1] dirp_out, double* free_path, double* p) nogil:

    cdef double pos_collision[3]
    cdef double pos_recoil[3]

    free_path[0] = MEAN_FREE_PATH

    # random p + phi
    #cdef double p = PMAX * sqrt(np.random.rand())
    #cdef double fi = 2 * M_PI * np.random.rand()
    #cdef double p = PMAX * sqrt(urand())
    #cdef double fi = 2 * M_PI * urand()
    p[0] = PMAX * sqrt(fast_rand())
    cdef double fi = 2 * M_PI * fast_rand()

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
    cdef double n = sqrt(dirp_out[0]*dirp_out[0] +
                         dirp_out[1]*dirp_out[1] +
                         dirp_out[2]*dirp_out[2])
    dirp_out[0] /= n
    dirp_out[1] /= n
    dirp_out[2] /= n


# Python-visible wrapper
cpdef get_recoil_position(double[::1] pos,
                          double[::1] dir,
                          double[::1] dirp_out):
    cdef double free_path, p
    get_recoil_position_c(pos, dir, dirp_out, &free_path, &p)
    return free_path, p