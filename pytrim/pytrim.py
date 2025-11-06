"""
PyTRIM main – multiprocessing ready
"""

from math import sqrt
import time
import numpy as np

from multiprocessing import Pool

import select_recoil
import scatter
import estop
import geometry
import trajectory



# ==========================================================
# global physics constants (so workers see them on import!)
# ==========================================================

# Initial conditions
pos_init = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dir_init = np.array([0.0, 0.0, 1.0], dtype=np.float64)
e_init = 20000.0

G_pos_init = pos_init
G_dir_init = dir_init
G_e_init   = e_init

zmin = 0.0
zmax = 4000.0
z1 = 5
m1 = 11.009
z2 = 14
m2 = 28.086
density = 0.04994
corr_lindhard = 1.5

# do setup ONCE at import time (workers import this file fresh!)
select_recoil.setup(density)
scatter.setup(z1, m1, z2, m2)
estop.setup(corr_lindhard, z1, m1, z2, density)
geometry.setup(zmin, zmax)
trajectory.setup()


def run_one(_):
    return trajectory.trajectory(G_pos_init, G_dir_init, G_e_init)


#main
def main(nion:int = 1000):

    start_time = time.time()

    # multiprocessingl
    with Pool() as pool:
        results = pool.map(run_one, range(nion))

    # accumulate
    count_inside = 0
    mean_z = 0.0
    std_z  = 0.0

    for pos, dir, e, is_inside in results:
        if is_inside:
            count_inside += 1
            mean_z += pos[2]
            std_z  += pos[2]**2

    if count_inside > 0:
        mean_z /= count_inside
        std_z   = sqrt(std_z/count_inside - mean_z**2)
    else:
        mean_z = 0.0
        std_z  = 0.0

    print(f"Number of ions stopped inside the target: {count_inside} / {nion}")
    print(f"Mean penetration depth: {mean_z:.2f} A")
    print(f"Std depth: {std_z:.2f} A")

    end_time = time.time()
    print(f"Simulation time: {end_time - start_time:.2f} s")


if __name__ == "__main__":
    main()