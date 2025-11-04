# geometry.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

cdef double ZMIN = 0.0
cdef double ZMAX = 0.0

def setup(double zmin, double zmax):
    global ZMIN, ZMAX
    ZMIN = zmin
    ZMAX = zmax

cpdef bint is_inside_target(double[:] pos):
    cdef double z = pos[2]
    return (ZMIN <= z) and (z <= ZMAX)