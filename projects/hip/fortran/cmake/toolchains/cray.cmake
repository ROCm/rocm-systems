# Cray toolchain. ftn and cc are the Cray compiler wrappers, which select the underlying compiler from the loaded PrgEnv module.
#
# Usage, from projects/hip/fortran:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/cray.cmake
#
# The compilers are looked up on PATH. Only the Fortran one actually matters
# here: the bindings are pure Fortran, and the C entry points they bind to come
# from the HIP runtime at link time.
#
# Free form and C preprocessing are requested by the CMakeLists via the
# Fortran_FORMAT and Fortran_PREPROCESS target properties, so CMake emits
# whichever flag this compiler expects and none is hardcoded here. No
# line-length flag is needed either: the generated source wraps at 112 columns,
# inside the 132 the free-form standard guarantees.

set(CMAKE_Fortran_COMPILER ftn CACHE FILEPATH "Fortran compiler")
set(CMAKE_C_COMPILER       cc CACHE FILEPATH "C compiler")
