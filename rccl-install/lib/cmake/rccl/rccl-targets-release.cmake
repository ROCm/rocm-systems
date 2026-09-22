#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "roc::rccl" for configuration "Release"
set_property(TARGET roc::rccl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(roc::rccl PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "rocprofiler-register::rocprofiler-register;hsa-runtime64::hsa-runtime64;rocm-core"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/librccl.so.1.0"
  IMPORTED_SONAME_RELEASE "librccl.so.1"
  )

list(APPEND _cmake_import_check_targets roc::rccl )
list(APPEND _cmake_import_check_files_for_roc::rccl "${_IMPORT_PREFIX}/lib/librccl.so.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
