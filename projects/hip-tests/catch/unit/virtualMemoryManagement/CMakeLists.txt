# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    hipMemGetAllocationGranularity.cc
    hipMemRetainAllocationHandle.cc)

if(HIP_PLATFORM MATCHES "nvidia")
  set(TEST_SRC
    ${TEST_SRC}
    hipMemMapArrayAsync.cc)
endif()

if(UNIX)
  if(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC
      ${TEST_SRC}
      hipMemGetHandleForAddressRange.cc)
  endif()
  set(TEST_SRC
      ${TEST_SRC}
      hipMemExportToShareableHandle.cc
      hipMemImportFromShareableHandle.cc)
endif()

if(HIP_PLATFORM MATCHES "amd")
  set(TEST_SRC
      ${TEST_SRC}
      hipGetProcAddressVmmApis.cc)
endif()

set(TEST_SRC
    ${TEST_SRC}
    hipMemAddressFree.cc
    hipMemAddressReserve.cc
    hipMemCreate.cc
    hipMemSetGetAccess.cc
    hipMemGetAllocationPropertiesFromHandle.cc
    hipMemMap.cc
    hipMemRelease.cc
    hipMemUnmap.cc)

# Verify MPI dev headers are actually present before calling find_package(MPI).
# A broken install (compiler wrapper present, but libopenmpi-dev headers missing)
# makes find_package(MPI)'s internal try_compile error out on a non-existent
# include path. We portably discover the include dir from the MPI compiler
# wrapper and confirm mpi.h exists on disk before trusting find_package.
find_program(MPI_CXX_WRAPPER NAMES mpicxx mpic++ mpiCC)
set(MPI_WRAPPER_INCLUDE_HINTS "")
if(MPI_CXX_WRAPPER)
  execute_process(
    COMMAND ${MPI_CXX_WRAPPER} --showme:incdirs
    OUTPUT_VARIABLE MPI_WRAPPER_INCDIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  separate_arguments(MPI_WRAPPER_INCLUDE_HINTS UNIX_COMMAND "${MPI_WRAPPER_INCDIRS}")
endif()
find_path(MPI_REAL_INCLUDE_DIR NAMES mpi.h HINTS ${MPI_WRAPPER_INCLUDE_HINTS})

set(MPI_USABLE FALSE)
if(UNIX AND MPI_REAL_INCLUDE_DIR)
  find_package(MPI QUIET)
  if(MPI_FOUND AND TARGET MPI::MPI_CXX)
    set(MPI_USABLE TRUE)
  endif()
endif()

if(MPI_USABLE)
  set(TEST_SRC
      ${TEST_SRC}
      hipMemFabricHandle.cc)
  set(VMM_LINKER_LIBS MPI::MPI_CXX)
  message(STATUS "MPI found and usable - including hipMemFabricHandle.cc")
else()
  message(STATUS "MPI not found or not usable - skipping hipMemFabricHandle.cc (install libopenmpi-dev to enable)")
endif()

hip_add_exe_to_target(NAME VirtualMemoryManagementTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME build_tests COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
  LINKER_LIBS ${VMM_LINKER_LIBS})
