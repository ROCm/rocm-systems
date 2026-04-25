# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(AMD_ICD_FOUND)
  return()
endif()

if(WIN32)
  find_path(AMD_ICD_LIBRARY_DIR OpenCL.lib
    HINTS
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../icd        #for clinfo
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../icd     #for ocltst
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../../icd  #for ocltst modules
    NO_DEFAULT_PATH)
else()
# The following line is used by AMD python script to build this repo in Linux.
#
# If people only use cmake to build this repo, ICD repo may not be built yet.
# As a result, ICD cannot be found because ICD does not exist.
# Take ocltst as an example how to build this repo under such scenario.
# ocltst cmake uses find_library command to find OpenCL. If find_library can't
# find ICD built by AMD, find_library command continue to search OpenCL
# installed in system. For example, find_library can find OpenCL installed by
# AMD rocm driver, /opt/rocm/lib/libOpenCL.so.
#
# Another solution is to build ICD loader first. Then install ICD loader into
# system. This solution is used by ROCM stack currently.

   find_path(AMD_ICD_LIBRARY_DIR libOpenCL.so
    HINTS
      # python build script
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../icd        #for clinfo
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../icd     #for ocltst
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../../icd  #for ocltst modules
      # rocm stack build scripts
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../OpenCL-ICD-Loader       #for clinfo
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../OpenCL-ICD-Loader    #for ocltst
      ${CMAKE_CURRENT_BINARY_DIR}/../../../../../../OpenCL-ICD-Loader #for ocltst modules
      # pure cmake method
      /opt/rocm/lib
      # centos/sles
      /usr/lib64
      /usr/lib
      # ubuntu
      /usr/lib/x86_64-linux-gnu
    NO_DEFAULT_PATH)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD_ICD
  "\nICD not found. Build may succeed if OpenCL ICD is installed in the system"
  AMD_ICD_LIBRARY_DIR)

mark_as_advanced(AMD_ICD_LIBRARY_DIR)
