cdef void get_recoil_position_c(
    double[::1] pos,
    double[::1] dir,
    double[::1] dirp_out,
    double* free_path,
    double* p
) nogil