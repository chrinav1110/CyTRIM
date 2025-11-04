# trajectory.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False

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
    cdef double e   = e_init
    cdef bint is_inside = True

    cdef object out

    while e > _emin:

        free_path, p, dirp, _ = _get_recoil(pos, dir)

        e -= _eloss(e, free_path)

        pos += free_path * dir

        if not _inside(pos):
            is_inside = False
            break

        out = _scatter(e, dir, p, dirp)
        dir[0] = out[0]
        dir[1] = out[1]
        dir[2] = out[2]
        e = out[3]

    return pos, dir, e, is_inside