Please consider the following for easier retrieval of hipFile project data:
hipFile is located at projects/hipfile
hipFile CI is located at .github/workflows/hipfile-*.yml
hipFile Python bindings are located at projects/hipfile/python

hipFile C Library
- Use projects/hipfile/build as the build directory
- Configure using `cmake -DCMAKE_CXX_COMPILER=amdclang++ ..`
- CMake build with `cmake --build . --parallel`
- Run unit tests from the build directory with `ctest -V -L "unit"`
- System & Stress tests may not work on all systems
- hipFile C Header to be found at projects/hipfile/include
- hipFile C library to be found at projects/hipfile/build/src/amd_detail
- projects/hipfile/util/format-source.sh is a clang-format wrapper for linting. Optionally takes the clang toolchain major version as an arg.

hipFile Python Bindings
- Uses Cython (see _chipfile.pxd & _hipfile.pyx)
- Locally, I have a Python venv setup at .venv. Please consider it if needing to test the Python code
- If needing to perform an install step for the Python bindings for test purposes, install it as an editable package within the .venv context
- Built by `scikit-build-core`
- If building/installing, be aware that the include & library paths for hipFile may need to be explicitly provided. In this case, use the following as a basis for your decision: 
  1. First, try `pip install -e python` — the CMakeLists.txt specifies sane defaults
  2. If that fails, add explicit paths: `pip install -e python -Ccmake.define.HIPFILE_INCLUDE_DIR=../include -Ccmake.define.HIPFILE_LIBRARY=../build/src/amd_detail -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include`
  3. Assumes CWD is projects/hipfile
- The Python code here uses a combination of 'pylint' and 'black' for linting
