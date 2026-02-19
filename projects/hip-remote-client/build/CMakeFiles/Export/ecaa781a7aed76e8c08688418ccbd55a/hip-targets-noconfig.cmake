#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hip::amdhip64" for configuration ""
set_property(TARGET hip::amdhip64 APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(hip::amdhip64 PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libamdhip64.1.0.0.dylib"
  IMPORTED_SONAME_NOCONFIG "@rpath/libamdhip64.1.dylib"
  )

list(APPEND _cmake_import_check_targets hip::amdhip64 )
list(APPEND _cmake_import_check_files_for_hip::amdhip64 "${_IMPORT_PREFIX}/lib/libamdhip64.1.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
