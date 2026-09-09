cmake_minimum_required(VERSION 3.16)

###############################################################################
# Standalone rocshmem4py build
#
# This is a self-contained Python project: `python -m build` (or a direct
# cmake invocation) drives this file as the top-level CMake project. We locate
# hipcc and pin CMAKE_CXX_COMPILER ourselves before project(), then build the
# nanobind extension against an installed rocSHMEM discovered via
# find_package(rocshmem CONFIG) (CMAKE_PREFIX_PATH). THEROCK_TOOLCHAIN_ROOT is
# honored as an alternate ROCm toolchain location.
###############################################################################
if(DEFINED ENV{THEROCK_TOOLCHAIN_ROOT} AND NOT THEROCK_TOOLCHAIN_ROOT)
  set(THEROCK_TOOLCHAIN_ROOT "$ENV{THEROCK_TOOLCHAIN_ROOT}" CACHE PATH "TheRock toolchain root")
endif()

if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm install directory")
elseif(THEROCK_TOOLCHAIN_ROOT)
  set(ROCM_PATH "${THEROCK_TOOLCHAIN_ROOT}" CACHE STRING "ROCm install directory")
else()
  set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm install directory")
endif()

find_program(HIPCC_EXECUTABLE hipcc
             PATHS
               ${ROCM_PATH}/bin
               ${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin
             NO_DEFAULT_PATH)

if(NOT HIPCC_EXECUTABLE)
  message(FATAL_ERROR "hipcc not found under ${ROCM_PATH}/bin or "
                      "${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin. "
                      "Set ROCM_PATH or THEROCK_TOOLCHAIN_ROOT.")
endif()

set(CMAKE_CXX_COMPILER "${HIPCC_EXECUTABLE}" CACHE FILEPATH "C++ compiler")
project(rocshmem4py LANGUAGES CXX)

# Single Python discovery for the whole file.  nanobind's CMake config
# requires the modern FindPython module with the Development.Module component
# (Development.Module is sufficient for building extension modules and avoids
# pulling in the embed/libpython requirement of the full Development component).
#
# Honor an explicitly provided interpreter so downstream builds pin the right
# Python: setuptools passes -DPYTHON_EXECUTABLE, and the TheRock super-build
# passes -DPython3_EXECUTABLE.  FindPython only consults Python_EXECUTABLE, so
# seed it from whichever hint was given.
if(NOT DEFINED Python_EXECUTABLE)
  if(DEFINED PYTHON_EXECUTABLE)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  elseif(DEFINED Python3_EXECUTABLE)
    set(Python_EXECUTABLE "${Python3_EXECUTABLE}")
  endif()
endif()
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}" CACHE FILEPATH "Python interpreter" FORCE)

message(STATUS "Python_VERSION: ${Python_VERSION}")
message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

###############################################################################
# Find Dependencies
###############################################################################

# APPEND, never PREPEND: a caller-supplied CMAKE_PREFIX_PATH (the rocSHMEM
# install, forwarded by setup.py as a -D cache var) must stay *ahead* of
# ROCM_PATH so find_package(rocshmem) resolves the intended install and not one
# that may also ship under /opt/rocm. ROCM_PATH is appended only so
# find_package(hip) can locate HIP.
list(APPEND CMAKE_PREFIX_PATH
     ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/lib/cmake/hip)
if(THEROCK_TOOLCHAIN_ROOT)
  list(APPEND CMAKE_PREFIX_PATH
       ${THEROCK_TOOLCHAIN_ROOT}
       ${THEROCK_TOOLCHAIN_ROOT}/lib/cmake/hip)
endif()
find_package(hip QUIET CONFIG
             HINTS ${ROCM_PATH}/lib/cmake/hip
                   ${THEROCK_TOOLCHAIN_ROOT}/lib/cmake/hip)
if(hip_FOUND)
  message(STATUS "HIP found: ${HIP_VERSION}")
else()
  message(STATUS "HIP package not found via find_package, using manual configuration")
  if(EXISTS "${ROCM_PATH}/include/hip")
    set(HIP_ROOT_DIR "${ROCM_PATH}")
  elseif(THEROCK_TOOLCHAIN_ROOT AND EXISTS "${THEROCK_TOOLCHAIN_ROOT}/include/hip")
    set(HIP_ROOT_DIR "${THEROCK_TOOLCHAIN_ROOT}")
  else()
    set(HIP_ROOT_DIR "${ROCM_PATH}")
  endif()
  set(HIP_INCLUDE_DIRS "${HIP_ROOT_DIR}/include")
  set(HIP_LIBRARIES "amdhip64")
  include_directories(${HIP_INCLUDE_DIRS})
  link_directories(${HIP_ROOT_DIR}/lib)
endif()

# Binding backend: nanobind.  It builds the `_rocshmem4py` module whose module
# name, function names, argument behavior, and return types are the stable
# public API contract.  (Python was already located above with the
# Development.Module component nanobind needs.)
#
# nanobind is pip-installed in the build environment, so ask the Python module
# for its CMake config directory and add it to the prefix path before
# find_package(nanobind).
execute_process(
  COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
  RESULT_VARIABLE _NB_CMAKE_DIR_OK
  OUTPUT_VARIABLE _NB_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(_NB_CMAKE_DIR_OK EQUAL 0 AND _NB_CMAKE_DIR)
  message(STATUS "nanobind CMake dir (pip): ${_NB_CMAKE_DIR}")
  list(APPEND CMAKE_PREFIX_PATH ${_NB_CMAKE_DIR})
endif()

find_package(nanobind CONFIG REQUIRED)
message(STATUS "nanobind found: ${nanobind_DIR}")

# Locate an installed rocSHMEM through its exported CMake package (installed at
# <prefix>/lib/cmake/rocshmem by rocm_install). find_package is the standard
# discovery mechanism and honors CMAKE_PREFIX_PATH, so pointing the build at a
# rocSHMEM install is just:
#     CMAKE_PREFIX_PATH=<rocshmem-install> ...
# ROCSHMEM_HOME is retained only as an optional convenience that seeds the
# prefix path; it is no longer required.
if(DEFINED ENV{ROCSHMEM_HOME})
  list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCSHMEM_HOME}")
endif()

# Minimum rocSHMEM (vendor/package) version this binding is API-compatible with.
# The binding wraps rocshmem_align / rocshmem_calloc / rocshmem_buffer_unregister_all,
# which were introduced in rocSHMEM 3.5.0 (see projects/rocshmem/CHANGELOG.md);
# building against an older install fails at compile time on the missing symbols.
# rocSHMEM's config-version file treats "installed >= requested" as compatible,
# so this accepts 3.5.0 and newer and rejects anything older with a clear error.
set(ROCSHMEM_MIN_VERSION 3.5.0)
find_package(rocshmem ${ROCSHMEM_MIN_VERSION} CONFIG REQUIRED)

set(ROCSHMEM_LINK_TARGET roc::rocshmem)
set(ROCSHMEM_INCLUDE_DIRS ${rocshmem_INCLUDE_DIRS})
message(STATUS "ROCSHMEM found via find_package: ${rocshmem_DIR}")
message(STATUS "  ROCSHMEM_INCLUDE_DIRS: ${ROCSHMEM_INCLUDE_DIRS}")

###############################################################################
# GPU architecture selection for device code objects
#
# rocSHMEM is compiled with relocatable device code (-fgpu-rdc), so the *final*
# device link that produces GPU code objects happens here, in this extension.
# Without an explicit --offload-arch, hipcc finalizes device code for a single
# default architecture (typically gfx942); the resulting wheel then aborts on
# any other GPU with hipErrorInvalidKernelFile (218). We therefore emit code
# objects for every architecture the linked rocSHMEM was built for.
#
# Default: use ROCSHMEM_OFFLOAD_TARGETS from the selected package's installed
# rocshmem_config.h. This records the architectures compiled into librocshmem.a
# and remains authoritative regardless of whether standalone JIT bitcode is
# installed. Override with a supported subset by passing
# -DROCSHMEM_GPU_TARGETS=gfx942,gfx950,...
#
# NOTE: we deliberately use our own ROCSHMEM_GPU_TARGETS cache variable rather
# than the ambient GPU_TARGETS: find_package(hip) above sets GPU_TARGETS to the
# *build host's* GPU, which would otherwise silently pin the wheel to a single
# arch and defeat the auto-detection.
set(ROCSHMEM_GPU_TARGETS "" CACHE STRING
    "GPU architectures to emit device code objects for (default: auto-detect from the rocSHMEM install)")

set(_ROCSHMEM_CONFIG_HEADER
    "${ROCSHMEM_INCLUDE_DIRS}/rocshmem/rocshmem_config.h")
if(NOT EXISTS "${_ROCSHMEM_CONFIG_HEADER}")
  message(FATAL_ERROR
    "rocshmem4py: selected rocSHMEM package does not provide "
    "${_ROCSHMEM_CONFIG_HEADER}")
endif()

file(READ "${_ROCSHMEM_CONFIG_HEADER}" _ROCSHMEM_CONFIG_CONTENTS)
string(REGEX MATCH
  "#[ \t]*define[ \t]+ROCSHMEM_OFFLOAD_TARGETS[ \t]+\"([^\"]+)\""
  _ROCSHMEM_TARGETS_DEFINITION
  "${_ROCSHMEM_CONFIG_CONTENTS}")
set(_ROCSHMEM_AVAILABLE_ARCHES "${CMAKE_MATCH_1}")
string(REGEX REPLACE "[ \t]+" ";"
  _ROCSHMEM_AVAILABLE_ARCHES "${_ROCSHMEM_AVAILABLE_ARCHES}")
if(NOT _ROCSHMEM_TARGETS_DEFINITION OR NOT _ROCSHMEM_AVAILABLE_ARCHES)
  message(FATAL_ERROR
    "rocshmem4py: ${_ROCSHMEM_CONFIG_HEADER} does not record any "
    "static-library GPU targets in ROCSHMEM_OFFLOAD_TARGETS")
endif()

set(_ROCSHMEM_ARCHES "${ROCSHMEM_GPU_TARGETS}")
string(REPLACE "," ";" _ROCSHMEM_ARCHES "${_ROCSHMEM_ARCHES}")
if(_ROCSHMEM_ARCHES)
  foreach(_arch IN LISTS _ROCSHMEM_ARCHES)
    list(FIND _ROCSHMEM_AVAILABLE_ARCHES "${_arch}" _arch_index)
    if(_arch_index EQUAL -1)
      message(FATAL_ERROR
        "rocshmem4py: ROCSHMEM_GPU_TARGETS requests ${_arch}, but the "
        "selected rocSHMEM package was built for: "
        "${_ROCSHMEM_AVAILABLE_ARCHES}")
    endif()
  endforeach()
else()
  set(_ROCSHMEM_ARCHES "${_ROCSHMEM_AVAILABLE_ARCHES}")
endif()
list(REMOVE_DUPLICATES _ROCSHMEM_ARCHES)

set(_ROCSHMEM_OFFLOAD_FLAGS "")
message(STATUS "rocshmem4py: building device code objects for: ${_ROCSHMEM_ARCHES}")
foreach(_arch IN LISTS _ROCSHMEM_ARCHES)
  list(APPEND _ROCSHMEM_OFFLOAD_FLAGS "--offload-arch=${_arch}")
endforeach()

###############################################################################
# Build Extension Module
###############################################################################

nanobind_add_module(_rocshmem4py NB_STATIC src/rocshmem4py.cc)
# rocSHMEM bindings throw std::runtime_error (translated to RuntimeError)
# and the team-config caster relies on RTTI.  nanobind defaults to
# -fno-rtti/-fno-exceptions, so re-enable them here and undo any global
# compile options that would otherwise disable exceptions on the static
# nanobind helper library.
target_compile_options(_rocshmem4py PRIVATE -frtti -fexceptions)
if(TARGET nanobind-static)
  get_target_property(_NB_OPTS nanobind-static COMPILE_OPTIONS)
  if(_NB_OPTS)
    list(REMOVE_ITEM _NB_OPTS "-fno-rtti" "-fno-exceptions")
    set_target_properties(nanobind-static PROPERTIES COMPILE_OPTIONS "${_NB_OPTS}")
  endif()
  target_compile_options(nanobind-static PRIVATE -frtti -fexceptions)
endif()

set_target_properties(_rocshmem4py PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(_rocshmem4py PRIVATE
  ${ROCSHMEM_INCLUDE_DIRS}
  ${Python_INCLUDE_DIRS}
  ${ROCM_PATH}/include
)
if(THEROCK_TOOLCHAIN_ROOT)
  target_include_directories(_rocshmem4py PRIVATE
    ${THEROCK_TOOLCHAIN_ROOT}/include
  )
endif()

# rocSHMEM's exported target conditionally references MPI::MPI_CXX when the
# selected package was built with external MPI. Its package config does not yet
# create that target, so retain discovery here. Do not link MPI directly:
# roc::rocshmem must be the authority on whether the extension requires MPI.
find_package(MPI QUIET)

if(hip_FOUND)
  target_link_libraries(_rocshmem4py PRIVATE
    ${ROCSHMEM_LINK_TARGET}
    hip::device
    hip::host
  )
else()
  target_link_libraries(_rocshmem4py PRIVATE
    ${ROCSHMEM_LINK_TARGET}
    amdhip64
    hsa-runtime64
  )
endif()

target_link_libraries(_rocshmem4py PRIVATE
  ${CMAKE_DL_LIBS}
  stdc++
  m
  pthread
)

target_compile_options(_rocshmem4py PRIVATE
  -D__HIP_PLATFORM_AMD__
  ${_ROCSHMEM_OFFLOAD_FLAGS}
)

target_link_options(_rocshmem4py PRIVATE
  -fgpu-rdc
  --hip-link
  ${_ROCSHMEM_OFFLOAD_FLAGS}
  -lamdhip64
  -lhsa-runtime64
)

# setuptools positions the .so via CMAKE_LIBRARY_OUTPUT_DIRECTORY; install(.)
# lands the extension at the wheel root for built distributions.
install(TARGETS _rocshmem4py LIBRARY DESTINATION .)
