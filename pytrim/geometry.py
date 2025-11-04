"""Target-geometry related operations.

Currently, only a planar target geometry is supported.

Available functions:
    setup: setup module variables.
    is_inside_target: check if a given position is inside the target
"""
"""
Target-geometry related operations.
"""

# optional pure-python mode access to cython decorators
try:
    import cython
except ImportError:
    cython = None


ZMIN: float = 0.0
ZMAX: float = 0.0


def setup(zmin: float, zmax: float) -> None:
    """
    Define the geometry of the target.
    """
    global ZMIN, ZMAX
    ZMIN = zmin
    ZMAX = zmax


@cython.locals(z=float)
def is_inside_target(pos) -> bool:
    """
    Check if a given position is inside the target.
    """
    # local caching → faster, and cythonable
    z = pos[2]
    return (ZMIN <= z) and (z <= ZMAX)




'''
def setup(zmin, zmax):
    """Define the geometry of the target.
    
    Parameters:
        zmin (int): minimum z coordinate of the target (A)
        zmax (int): maximum z coordinate of the target (A)

    Returns:
        None
    """
    global ZMIN, ZMAX

    ZMIN = zmin
    ZMAX = zmax


def is_inside_target(pos):
    """Check if a given position is inside the target.

    Parameters:
        pos (ndarray): position to check (size 3)

    Returns:
        bool: True if position is inside the target, False otherwise
    """
    if ZMIN <= pos[2] <= ZMAX:
        return True
    else:
        return False 
        
'''