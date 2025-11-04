from setuptools import setup
from Cython.Build import cythonize
import numpy as np
'''
setup(
    ext_modules = cythonize([
        #"select_recoil.pyx",
        #"scatter.pyx",
        "trajectory.pyx",
        # pure python modules can be left out entirely
    ], language_level="3"),
)'''



setup(
    ext_modules=cythonize([
        "trajectory.pyx",
        "select_recoil.pyx",
        "scatter.pyx",
        "estop.pyx",
        "geometry.pyx"
        ],
        language_level="3",
        compiler_directives={"boundscheck": False, "wraparound": False,
                             "cdivision": True,
                             "profile": True
                             },
    ),
    include_dirs=[np.get_include()],
)