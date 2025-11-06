# geometry.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: profile=True

"""Target-geometry related operations.

Currently, only a planar target geometry is supported.

Available functions:
    setup: setup module variables.
    is_inside_target: check if a given position is inside the target
"""
"""
Target-geometry related operations.
"""

cdef double ZMIN = 0.0
cdef double ZMAX = 0.0

def setup(double zmin, double zmax):
    global ZMIN, ZMAX
    ZMIN = zmin
    ZMAX = zmax

cpdef bint is_inside_target(double[:] pos):
    cdef double z = pos[2]
    return (ZMIN <= z) and (z <= ZMAX)