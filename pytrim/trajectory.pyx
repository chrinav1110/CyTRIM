# trajectory.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False
# cython: profile=True

# we import python-level functions (cython compiled versions available)
from select_recoil import get_recoil_position
from scatter import scatter
from estop import eloss
from geometry import is_inside_target

import numpy as np

cdef double EMIN = 5.0


def setup():
    global EMIN
    EMIN = 5.0


cpdef trajectory(object pos_init, object dir_init, double e_init):
    """
    Simulate one trajectory.
    This uses python operations intentionally (Numpy arrays) = identical logic.
    """
    # local aliases (saves repeated globals lookups)
    cdef object _get_recoil = get_recoil_position
    cdef object _scatter    = scatter
    cdef object _eloss      = eloss
    cdef object _inside     = is_inside_target
    cdef double _emin       = EMIN

    # copy to avoid alias issues
    cdef object pos = pos_init.copy()
    cdef object dir = dir_init.copy()

    cdef double[:] posv = pos
    cdef double[:] dirv = dir

    cdef double e   = e_init
    cdef bint is_inside = True

    cdef object out = np.empty(8, dtype=np.float64)
    cdef double[:] outv = out    # memoryview of the same buffer

    cdef object dirp = np.empty(3, dtype=np.float64)
    cdef double[:] dirpv = dirp

    while e > _emin:

        free_path, p = _get_recoil(pos, dir, dirpv)

        e -= _eloss(e, free_path)

        #pos += free_path * dir
        posv[0] += free_path * dirv[0]
        posv[1] += free_path * dirv[1]
        posv[2] += free_path * dirv[2]

        if not _inside(pos):
            is_inside = False
            break

        _scatter(e, dir, p, dirp, outv)
        dir[0] = outv[0]
        dir[1] = outv[1]
        dir[2] = outv[2]
        e = out[3]

    return pos, dir, e, is_inside