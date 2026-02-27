cmake_minimum_required(VERSION 3.16)
project(rocshmem4py LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python REQUIRED)

message(STATUS "Python3_VERSION: ${Python3_VERSION}")

###############################################################################
# ROCM/HIP Configuration
###############################################################################
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm install directory")
else()
  set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm install directory")
endif()

set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
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

# Find HIP
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/lib/cmake/hip)
find_package(hip QUIET CONFIG HINTS ${ROCM_PATH}/lib/cmake/hip)
if(hip_FOUND)
  message(STATUS "HIP found: ${HIP_VERSION}")
else()
  message(STATUS "HIP package not found via find_package, using manual configuration")
  set(HIP_INCLUDE_DIRS "${ROCM_PATH}/include")
  set(HIP_LIBRARIES "amdhip64")
  include_directories(${HIP_INCLUDE_DIRS})
  link_directories(${ROCM_PATH}/lib)
endif()

# Find pybind11
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c 
          "from __future__ import print_function; import os; import pybind11; print(os.path.dirname(pybind11.__file__), end='')"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PYBIND11_DIR
)

if(NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
endif()

message(STATUS "pybind11 directory: ${PYBIND11_DIR}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)

# Find ROCSHMEM
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
  message(STATUS "ROCSHMEM found!")
  message(STATUS "  ROCSHMEM_INCLUDE_DIRS: ${ROCSHMEM_INCLUDE_DIRS}")
  message(STATUS "  ROCSHMEM_LIBRARIES: ${ROCSHMEM_LIBRARIES}")
else()
  message(FATAL_ERROR "ROCSHMEM not found. Set ROCSHMEM_HOME environment variable.")
endif()

###############################################################################
# Build Extension Module
###############################################################################

pybind11_add_module(_rocshmem4py src/rocshmem4py.cc)

set_target_properties(_rocshmem4py PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(_rocshmem4py PRIVATE
  ${ROCSHMEM_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${ROCM_PATH}/include
)

find_package(MPI QUIET)

if(hip_FOUND)
  target_link_libraries(_rocshmem4py PRIVATE
    ${ROCSHMEM_LIBRARIES}
    hip::device
    hip::host
  )
else()
  target_link_libraries(_rocshmem4py PRIVATE
    ${ROCSHMEM_LIBRARIES}
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
  $<$<COMPILE_LANGUAGE:CXX>:-fno-gpu-rdc>
)

target_link_options(_rocshmem4py PRIVATE
  -fgpu-rdc
  --hip-link
  -lamdhip64
  -lhsa-runtime64
)

install(TARGETS _rocshmem4py
  LIBRARY DESTINATION .
)
