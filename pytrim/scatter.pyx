# scatter.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: profile=True

from libc.math cimport sqrt, exp, pow, fabs
import numpy as np
cimport numpy as np

# -------- module globals (set in setup) --------
cdef double ENORM = 1.0
cdef double RNORM = 1.0
cdef double DIRFAC = 1.0
cdef double DENFAC = 1.0

def setup(double z1, double m1, double z2, double m2):
    global ENORM, RNORM, DIRFAC, DENFAC
    cdef double m1_m2 = m1 / m2
    RNORM = 0.4685 / (z1**0.23 + z2**0.23)
    ENORM = 14.39979 * z1 * z2 / RNORM * (1.0 + m1_m2)
    DIRFAC = 2.0 / (1.0 + m1_m2)
    DENFAC = 4.0 * m1_m2 / ((1.0 + m1_m2) * (1.0 + m1_m2))

# -------- ZBL screening constants --------
cdef double A1=0.18175, A2=0.50986, A3=0.28022, A4=0.02817
cdef double B1=3.1998,  B2=0.94229, B3=0.4029,  B4=0.20162
cdef double A1B1=A1*B1, A2B2=A2*B2, A3B3=A3*B3, A4B4=A4*B4

cdef inline void ZBLscreen(double r, double* screen, double* dscreen) nogil:
    cdef double e1 = exp(-B1*r)
    cdef double e2 = exp(-B2*r)
    cdef double e3 = exp(-B3*r)
    cdef double e4 = exp(-B4*r)
    screen[0]  = A1*e1 + A2*e2 + A3*e3 + A4*e4
    dscreen[0] = -(A1B1*e1 + A2B2*e2 + A3B3*e3 + A4B4*e4)

# -------- apsis estimation constants --------
cdef double K2 = 0.38
cdef double K3 = 7.2
cdef double K1 = 1.0/(4.0*K2)
cdef double R12sq = (2.0*K2)*(2.0*K2)
cdef double R23sq = K3 / K2
cdef int    NITER = 1

cdef inline double estimate_apsis(double e, double p) nogil:
    cdef double psq = p*p
    cdef double r0sq = 0.5 * (psq + sqrt(psq*psq + 4.0*K3/e))
    cdef double r0, scr, dscr, num, den, resid

    if r0sq < R23sq:
        r0sq = psq + K2/e
        if r0sq < R12sq:
            r0 = (1.0 + sqrt(1.0 + 4.0*e*(e+K1)*psq)) / (2.0*(e+K1))
        else:
            r0 = sqrt(r0sq)
    else:
        r0 = sqrt(r0sq)

    # Newton iterations (exact algebra as Python version)
    for _ in range(NITER):
        ZBLscreen(r0, &scr, &dscr)
        num = r0*(r0 - scr/e) - psq
        den = 2.0*r0 - (scr + r0*dscr)/e
        r0 -= num/den

        resid = 1.0 - scr/(e*r0) - psq/(r0*r0)
        if fabs(resid) < 1e-4:
            break

    return r0

# -------- magic formula constants --------
cdef double C1=0.99229
cdef double C2=0.011615
cdef double C3=0.007122
cdef double C4=14.813
cdef double C5=9.3066

cdef inline double magic(double e, double p) nogil:
    cdef double r0 = estimate_apsis(e, p)
    cdef double scr, dscr, rho, sqe, alpha, beta, gamma, a, g, delta, c

    ZBLscreen(r0, &scr, &dscr)

    rho = 2.0*(e*r0 - scr) / (scr/r0 - dscr)
    sqe = sqrt(e)
    alpha = 1.0 + C1/sqe
    beta  = (C2 + sqe) / (C3 + sqe)
    gamma = (C4 + e)  / (C5 + e)
    a = 2.0 * alpha * e * pow(p, beta)
    g = gamma / (sqrt(1.0 + a*a) - a)
    delta = a * (r0 - p) / (1.0 + g)

    c = (p + rho + delta) / (r0 + rho)
    return c  # no clamp (Python parity)

# -------- main scatter --------
def scatter(double e, object dir, double p, object dirp):
    """
    Fast path: C math for everything (+ packed return).
    Assumes dir/dirp are contiguous float64 arrays of length 3.
    """
    cdef double[:] v  = dir   # projectile direction
    cdef double[:] rp = dirp  # impact-parameter direction

    # cos(theta/2) in CM (pure C math path)
    cdef double c_half
    with nogil:
        c_half = magic(e/ENORM, p/RNORM)

    # sin/cos psi in lab formula
    cdef double sin_psi = c_half
    cdef double t = 1.0 - sin_psi*sin_psi
    if t < 0.0:
        t = 0.0
    cdef double cos_psi = sqrt(t)

    # compute (cos_psi*dir + sin_psi*dirp)
    cdef double vec0 = cos_psi*v[0] + sin_psi*rp[0]
    cdef double vec1 = cos_psi*v[1] + sin_psi*rp[1]
    cdef double vec2 = cos_psi*v[2] + sin_psi*rp[2]

    # recoil direction (before normalization)
    cdef double rec0 = DIRFAC * cos_psi * vec0
    cdef double rec1 = DIRFAC * cos_psi * vec1
    cdef double rec2 = DIRFAC * cos_psi * vec2

    # new projectile direction (before normalization)
    cdef double new0 = v[0] - rec0
    cdef double new1 = v[1] - rec1
    cdef double new2 = v[2] - rec2

    # normalize new dir
    cdef double n = sqrt(new0*new0 + new1*new1 + new2*new2)
    if n != 0.0:
        new0 /= n; new1 /= n; new2 /= n
    else:
        new0 = v[0]; new1 = v[1]; new2 = v[2]

    # normalize recoil dir
    n = sqrt(rec0*rec0 + rec1*rec1 + rec2*rec2)
    if n != 0.0:
        rec0 /= n; rec1 /= n; rec2 /= n
    else:
        rec0 = v[0]; rec1 = v[1]; rec2 = v[2]

    # energies
    cdef double e_recoil = DENFAC * e * (1.0 - c_half*c_half)
    cdef double e_after  = e - e_recoil

    # packed output (8 doubles)
    cdef object out_arr = np.empty(8, dtype=np.float64)
    cdef double[:] out = out_arr
    out[0] = new0
    out[1] = new1
    out[2] = new2
    out[3] = e_after
    out[4] = rec0
    out[5] = rec1
    out[6] = rec2
    out[7] = e_recoil
    return out_arr
