# CyTRIM
A Python/Cython implementation of TRIM.

## Directories:
    pytrim: Python and Cython code 

##
    change folder: cd CyTRIM/pytrim
    compile with:  python .\buildCythonized.py
    call:  python -c "import cytrim; cytrim.main(N_Ions, use_multiprocessing)"
    for benchmark :  python -c "import cytrim; cytrim.benchmark('Test Name', use_multiprocessing)"
    with profiler, replace Nions (multiprocessing is disabled): 
    python -c "import cProfile, pstats; p=cProfile.Profile(); p.runcall(lambda: __import__('cytrim').main(N_Ions, 0)); s=pstats.Stats(p); s.sort_stats('tottime').print_stats()" > profilerLog.txt




