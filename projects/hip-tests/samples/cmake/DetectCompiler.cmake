# DetectCompiler.cmake
# Shared compiler detection logic for HIP samples
# 
# This module detects and sets CMAKE_CXX_COMPILER to amdclang++
# if ROCM_PATH is defined and no compiler is already set.
#
# Usage: include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/DetectCompiler.cmake)
#        (or appropriate relative path from sample directory)

# Dynamic compiler detection - set before project()
# Only applies if ROCM_PATH is defined and compiler not already set
if(NOT CMAKE_CXX_COMPILER AND DEFINED ROCM_PATH)
  if(WIN32)
    # Windows: Search recursively for amdclang++.exe (TheRock variants)
    file(GLOB_RECURSE HIP_CLANG_COMPILER "${ROCM_PATH}/../*/bin/amdclang++.exe")
    if(HIP_CLANG_COMPILER)
      list(GET HIP_CLANG_COMPILER 0 HIP_CLANG_COMPILER)
      message(STATUS "DetectCompiler: Found amdclang++: ${HIP_CLANG_COMPILER}")
    endif()
  else()
    # Linux: Try standard paths
    find_program(HIP_CLANG_COMPILER
      NAMES amdclang++ clang++
      PATHS ${ROCM_PATH}/llvm/bin ${ROCM_PATH}/bin
      NO_DEFAULT_PATH
    )
    if(HIP_CLANG_COMPILER)
      message(STATUS "DetectCompiler: Found amdclang++: ${HIP_CLANG_COMPILER}")
    endif()
  endif()
  
  if(HIP_CLANG_COMPILER)
    set(CMAKE_CXX_COMPILER ${HIP_CLANG_COMPILER})
  endif()
endif()

