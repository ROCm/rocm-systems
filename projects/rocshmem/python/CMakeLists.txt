cmake_minimum_required(VERSION 3.16)

###############################################################################
# Build mode: Standalone vs Sub-project
#
# Standalone:  pip install / direct cmake on python/. This CMakeLists is the
#              top-level project; we must locate hipcc and pin
#              CMAKE_CXX_COMPILER ourselves before calling project().
# Sub-project: added via add_subdirectory() from the parent rocSHMEM CMake
#              (TheRock super-build). The parent has already run project()
#              with the right toolchain, so we skip compiler selection.
#
###############################################################################
set(ROCSHMEM4PY_IS_TOP_LEVEL FALSE)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(ROCSHMEM4PY_IS_TOP_LEVEL TRUE)
endif()

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

if(ROCSHMEM4PY_IS_TOP_LEVEL)
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
endif()

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
# nanobind discovery is two-tiered:
#   1. Standalone (pip) builds: nanobind is pip-installed, so ask the Python
#      module for its CMake config directory and add it to the prefix path.
#   2. TheRock super-build: nanobind is provided as a CMake subproject
#      (therock-nanobind), so `python -m nanobind` is unavailable -- the
#      execute_process fails harmlessly and find_package resolves the
#      subproject-provided package config instead.
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

# Sub-project: the parent CMake exposes the in-tree rocshmem target.
# Standalone: locate an installed rocSHMEM via ROCSHMEM_HOME.
if(TARGET rocshmem)
  set(ROCSHMEM_FOUND TRUE)
  set(ROCSHMEM_LINK_TARGET roc::rocshmem)
  set(ROCSHMEM_INCLUDE_DIRS
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_BINARY_DIR}/include)
  message(STATUS "ROCSHMEM target found in parent project: ${ROCSHMEM_LINK_TARGET}")
else()
  if(DEFINED ENV{ROCSHMEM_HOME})
    set(ROCSHMEM_HOME "$ENV{ROCSHMEM_HOME}" CACHE STRING "ROCSHMEM install directory")
    list(APPEND CMAKE_PREFIX_PATH ${ROCSHMEM_HOME})
  endif()

  find_path(ROCSHMEM_INCLUDE_DIR
    NAMES rocshmem/rocshmem.hpp
    PATHS
      ${ROCSHMEM_HOME}
      ${ROCSHMEM_HOME}/../include
      /opt/rocm
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
  )

  find_library(ROCSHMEM_LIBRARY
    NAMES rocshmem
    PATHS
      ${ROCSHMEM_HOME}
      ${ROCSHMEM_HOME}/lib
      /opt/rocm
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
  )

  if(ROCSHMEM_INCLUDE_DIR AND ROCSHMEM_LIBRARY)
    set(ROCSHMEM_FOUND TRUE)
    set(ROCSHMEM_INCLUDE_DIRS ${ROCSHMEM_INCLUDE_DIR})
    if(EXISTS "${ROCSHMEM_HOME}/include/rocshmem")
      list(APPEND ROCSHMEM_INCLUDE_DIRS "${ROCSHMEM_HOME}/include/rocshmem")
    endif()
    if(EXISTS "${ROCSHMEM_HOME}/include")
      list(APPEND ROCSHMEM_INCLUDE_DIRS "${ROCSHMEM_HOME}/include")
    endif()
    set(ROCSHMEM_LIBRARIES ${ROCSHMEM_LIBRARY})
    set(ROCSHMEM_LINK_TARGET ${ROCSHMEM_LIBRARIES})
    message(STATUS "ROCSHMEM found!")
    message(STATUS "  ROCSHMEM_INCLUDE_DIRS: ${ROCSHMEM_INCLUDE_DIRS}")
    message(STATUS "  ROCSHMEM_LIBRARIES: ${ROCSHMEM_LIBRARIES}")
  else()
    message(FATAL_ERROR "ROCSHMEM not found. Set ROCSHMEM_HOME environment variable.")
  endif()
endif()

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

if(MPI_FOUND)
  target_link_libraries(_rocshmem4py PRIVATE ${MPI_CXX_LIBRARIES})
  target_include_directories(_rocshmem4py PRIVATE ${MPI_CXX_INCLUDE_DIRS})
endif()

target_link_libraries(_rocshmem4py PRIVATE
  ${CMAKE_DL_LIBS}
  stdc++
  m
  pthread
)

target_compile_options(_rocshmem4py PRIVATE
  -D__HIP_PLATFORM_AMD__
)

target_link_options(_rocshmem4py PRIVATE
  -fgpu-rdc
  --hip-link
  -lamdhip64
  -lhsa-runtime64
)

if(ROCSHMEM4PY_IS_TOP_LEVEL)
  # Standalone: setuptools positions the .so via CMAKE_LIBRARY_OUTPUT_DIRECTORY;
  # install(.) lands the extension at the wheel root for built distributions.
  install(TARGETS _rocshmem4py LIBRARY DESTINATION .)
else()
  # Sub-project (TheRock super-build): ship the binding under the rocSHMEM
  # subtree alongside lib/*.bc, lib/cmake/rocshmem, share/rocshmem.
  # Consumers add <prefix>/lib/rocshmem/python to PYTHONPATH.
  include(GNUInstallDirs)
  set(_ROCSHMEM4PY_DEST "${CMAKE_INSTALL_LIBDIR}/rocshmem/python/rocshmem4py")
  install(TARGETS _rocshmem4py LIBRARY DESTINATION ${_ROCSHMEM4PY_DEST})
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rocshmem4py/
    DESTINATION ${_ROCSHMEM4PY_DEST}
    FILES_MATCHING PATTERN "*.py"
  )
endif()
