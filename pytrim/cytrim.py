"""
CyTRIM main – multiprocessing
"""

from math import sqrt
import time
import numpy as np
import sys

from multiprocessing import Pool

import select_recoil
import scatter
import estop
import geometry
import trajectory

#only for benchmarking
import csv
import os

# ---------------------------------------------------------
# Benchmark
# ---------------------------------------------------------
ION_SETS = [1000]
RUNS = 10
RESULT_FILE = "benchmark_results.csv"


# Initial conditions
pos_init = np.array([0.0, 0.0, 0.0], dtype=np.float64)
dir_init = np.array([0.0, 0.0, 1.0], dtype=np.float64)
e_init = 50000.0

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

select_recoil.setup(density)
scatter.setup(z1, m1, z2, m2)
estop.setup(corr_lindhard, z1, m1, z2, density)
geometry.setup(zmin, zmax)
trajectory.setup()


def run_one(_):
    return trajectory.trajectory(G_pos_init, G_dir_init, G_e_init)


#main
#enable or disable multiprocessing
def main(nion=1000, use_mp=True):

    start_time = time.time()

    if not use_mp:
        # run single-threaded
        results = [run_one(i) for i in range(nion)]
    else:
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

    dt = time.time() - start_time
    print(f"Simulation time: {dt:.4f} s")

    return dt


def benchmark(name="default", use_mp = True):
    print("\n===== CyTRIM Benchmark =====")
    print(f"Benchmark name: {name}\n")

    # create CSV if not exists
    new_file = not os.path.exists(RESULT_FILE)
    with open(RESULT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["benchmark_name", "ions", "mean_time"])

        for nion in ION_SETS:
            print(f"--- nion = {nion} ---")
            times = []

            for r in range(RUNS):
                dt = main(nion, use_mp)
                times.append(dt)
                print(f"Run {r+1}: {dt:.4f} s")

            avg = sum(times) / RUNS
            print(f"Average for nion={nion}: {avg:.4f} s\n")

            # write to file
            writer.writerow([name, nion, avg])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        # optional: allow "python cytrim.py 500" → run main(500)
        if len(sys.argv) > 1:
            main(int(sys.argv[1]))
        else:
            main()