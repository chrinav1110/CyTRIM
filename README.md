# CyTRIM
A Python/Cython implementation of TRIM.

## Directories:
    pytrim: pure Python code, kept for reference.
    cytrim: Cython code ("model")
    gui: (Most of) the GUI code ("view")
    doc: Documentation

##
    change folder: cd CyTRIM/pytrim
    compile with:  python .\buildCythonized.py
    call:  python -c "import cytrim; cytrim.main(N_Ions)"
    for benchmark :  python -c "import cytrim; cytrim.benchmark('Test Name')"


