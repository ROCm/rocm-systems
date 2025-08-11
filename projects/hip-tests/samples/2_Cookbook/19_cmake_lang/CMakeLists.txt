# The test must be with CMake 3.18 or above which supports LINK_LANGUAGE that is necessary for
# Fortran build.
cmake_minimum_required(VERSION 3.18.0)
if(ENABLE_ADDRESS_SANITIZER)
  message(STATUS "This fortran sample is temporarily disabled due to SWDEV-462616.")
  return()
endif()
project(cmake_lang_test CXX Fortran)

if(UNIX)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
    endif()
  endif()
  # Search for rocm in common locations
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/llvm/bin/clang++")

find_package(hip REQUIRED)

if(TARGET build_cookbook)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(test_fortran ${EXCLUDE_OPTION} TestFortran.F90)
add_executable(test_cpp1 ${EXCLUDE_OPTION} MatrixTranspose.cpp)

target_link_libraries(test_cpp1 PUBLIC hip::device)

target_include_directories(test_cpp1 PRIVATE ../../common)

# Assuming to build a C/C++-to-Fortran library binding.
target_link_libraries(test_fortran PUBLIC hip::device)

if(TARGET build_cookbook)
add_dependencies(build_cookbook test_cpp1 test_fortran)
endif()
