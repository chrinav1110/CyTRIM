# scatter.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

"""Treat the scattering of a projectile on a target atom.

Currently, only the ZBL potential (Ziegler, Biersack, Littmark,
The Stopping and Range of Ions in Matter, Pergamon Press, 1985) is
implemented, along with Biersack's "magic formula" for the scattering
angle.

Available functions:
    setup: setup module variables.
    scatter: treat a scattering event.
"""

from libc.math cimport sqrt, exp, pow, fabs

# module globals
cdef double ENORM = 1.0
cdef double RNORM = 1.0
cdef double DIRFAC = 1.0
cdef double DENFAC = 1.0

def setup(double z1, double m1, double z2, double m2):
    global ENORM, RNORM, DIRFAC, DENFAC

    cdef double m1_m2 = m1 / m2
    RNORM = 0.4685 / (z1**0.23 + z2**0.23)
    ENORM = 14.39979 * z1 * z2 / RNORM * (1 + m1_m2)
    DIRFAC = 2 / (1 + m1_m2)
    DENFAC = 4 * m1_m2 / (1 + m1_m2)**2

# --- ZBL constants ---
cdef double A1 = 0.18175
cdef double A2 = 0.50986
cdef double A3 = 0.28022
cdef double A4 = 0.02817
cdef double B1 = 3.1998
cdef double B2 = 0.94229
cdef double B3 = 0.4029
cdef double B4 = 0.20162
cdef double A1B1 = A1 * B1
cdef double A2B2 = A2 * B2
cdef double A3B3 = A3 * B3
cdef double A4B4 = A4 * B4

cdef inline void ZBLscreen_vals(double r, double* screen, double* dscreen) nogil:
    cdef double e1 = exp(-B1 * r)
    cdef double e2 = exp(-B2 * r)
    cdef double e3 = exp(-B3 * r)
    cdef double e4 = exp(-B4 * r)
    screen[0]  = A1*e1 + A2*e2 + A3*e3 + A4*e4
    dscreen[0] = - (A1B1*e1 + A2B2*e2 + A3B3*e3 + A4B4*e4)

# --- apsis constants ---
cdef double K2    = 0.38
cdef double K3    = 7.2
cdef double K1    = 1.0/(4.0*K2)
cdef double R12sq = (2.0*K2)*(2.0*K2)
cdef double R23sq = K3 / K2

#Loop commented out because NITER set to 1
cdef int    NITER = 1


cdef inline double estimate_apsis(double e, double p) nogil:
    cdef double psq   = p*p
    cdef double r0sq  = 0.5 * (psq + sqrt(psq*psq + 4.0*K3/e))
    cdef double r0

    if r0sq < R23sq:
        r0sq = psq + K2/e
        if r0sq < R12sq:
            r0 = (1.0 + sqrt(1.0 + 4.0*e*(e+K1)*psq)) / (2.0*(e+K1))
        else:
            r0 = sqrt(r0sq)
    else:
        r0 = sqrt(r0sq)

    cdef double screen, dscreen
    cdef double numerator, denominator

    # Iter is set to 1, in that case the loop is not necessary
    '''
    cdef double residuum
    cdef int it
    for it in range(NITER):
        ZBLscreen_vals(r0, &screen, &dscreen)
        numerator   = r0*(r0 - screen/e) - psq
        denominator = 2.0*r0 - (screen + r0*dscreen)/e
        r0 -= numerator / denominator

        residuum = 1.0 - screen/(e*r0) - psq/(r0*r0)
        if fabs(residuum) < 1e-4:
            break
    '''
    numerator = r0 * (r0 - screen / e) - psq
    denominator = 2.0 * r0 - (screen + r0 * dscreen) / e
    r0 -= numerator / denominator

    return r0


# --- magic constants ---
cdef double C1 = 0.99229
cdef double C2 = 0.011615
cdef double C3 = 0.007122
cdef double C4 = 14.813
cdef double C5 = 9.3066

cdef inline double magic(double e, double p) nogil:
    cdef double r0 = estimate_apsis(e, p)
    cdef double screen, dscreen
    ZBLscreen_vals(r0, &screen, &dscreen)

    cdef double rho   = 2.0*(e*r0 - screen) / (screen/r0 - dscreen)
    cdef double sqrte = sqrt(e)
    cdef double alpha = 1.0 + C1/sqrte
    cdef double beta  = (C2 + sqrte) / (C3 + sqrte)
    cdef double gamma = (C4 + e) / (C5 + e)
    cdef double a     = 2.0 * alpha * e * pow(p, beta)
    cdef double g     = gamma / (sqrt(1.0 + a*a) - a)
    cdef double delta = a * (r0 - p) / (1.0 + g)

    cdef double cos_half_theta = (p + rho + delta) / (r0 + rho)
    if cos_half_theta > 1.0:
        cos_half_theta = 1.0
    elif cos_half_theta < -1.0:
        cos_half_theta = -1.0
    return cos_half_theta


cpdef scatter(double e, double[::1] dir, double p, double[::1] dirp, double[::1] out):

    cdef double cos_half_theta = magic(e/ENORM, p/RNORM)
    cdef double sin_psi = cos_half_theta
    cdef double cos_psi = sqrt(1 - sin_psi*sin_psi)

    cdef double v0 = cos_psi*dir[0] + sin_psi*dirp[0]
    cdef double v1 = cos_psi*dir[1] + sin_psi*dirp[1]
    cdef double v2 = cos_psi*dir[2] + sin_psi*dirp[2]

    cdef double rec0 = DIRFAC * cos_psi * v0
    cdef double rec1 = DIRFAC * cos_psi * v1
    cdef double rec2 = DIRFAC * cos_psi * v2

    cdef double new0 = dir[0] - rec0
    cdef double new1 = dir[1] - rec1
    cdef double new2 = dir[2] - rec2

    cdef double nnew = sqrt(new0*new0 + new1*new1 + new2*new2)
    if nnew != 0:
        new0/=nnew; new1/=nnew; new2/=nnew

    cdef double nrec = sqrt(rec0*rec0 + rec1*rec1 + rec2*rec2)
    if nrec != 0:
        rec0/=nrec; rec1/=nrec; rec2/=nrec

    cdef double e_recoil = DENFAC * e * (1 - cos_half_theta*cos_half_theta)
    cdef double e_after  = e - e_recoil

    out[0] = new0
    out[1] = new1
    out[2] = new2
    out[3] = e_after
    out[4] = rec0
    out[5] = rec1
    out[6] = rec2
    out[7] = e_recoil
    return
