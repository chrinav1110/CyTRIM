# estop.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

"""Calculate the electronic stopping power.

Currently, only the Lindhard model (Phys. Rev. 124, (1961) 128) with
a correction factor is implemented.

Available functions:
    setup: setup module variables.
    eloss: calculate the electronic energy loss.
"""

from libc.math cimport sqrt

cdef double FAC_LINDHARD = 0.0
cdef double DENSITY      = 0.0

def setup(double corr_lindhard, double z1, double m1, double z2, double density):

    global FAC_LINDHARD, DENSITY

    FAC_LINDHARD = corr_lindhard * 1.212 * z1**(7.0/6.0) * z2 / (
        (z1**(2.0/3.0) + z2**(2.0/3.0))**(3.0/2.0) * sqrt(m1) )
    DENSITY = density

cpdef double eloss(double e, double free_path):
    cdef double dee = FAC_LINDHARD * DENSITY * sqrt(e) * free_path
    if dee > e:
        dee = e
    return dee
