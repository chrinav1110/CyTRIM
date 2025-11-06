import os
import glob
import shutil
import subprocess

#Call using: python .\buildCythonized.py
#buildCythonized.py calls cythonized_setup.py


# ----------------------------------------------------
# go to the folder where THIS file is located
# so relative paths point to the right place
# ----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# ----------------------------------------------------
# clean old build artefacts
# ----------------------------------------------------
if os.path.exists("build"):
    shutil.rmtree("build")

for f in glob.glob("*.c"):
    os.remove(f)

for f in glob.glob("*.so") + glob.glob("*.pyd"):
    os.remove(f)

# ----------------------------------------------------
# run cython build
# ----------------------------------------------------
cython_setup = os.path.join(HERE, "cythonized_setup.py")

try:
    subprocess.check_call(["python", cython_setup, "build_ext", "--inplace"])
except subprocess.CalledProcessError as e:
    print("### build failed ###")
    print("command:", e.cmd)
    print("return code:", e.returncode)
    raise
