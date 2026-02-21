#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hip::amdhip64" for configuration "Debug"
set_property(TARGET hip::amdhip64 APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(hip::amdhip64 PROPERTIES
  IMPORTED_IMPLIB_DEBUG "${_IMPORT_PREFIX}/lib/amdhip64_7.lib"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/bin/amdhip64_7.dll"
  )

list(APPEND _cmake_import_check_targets hip::amdhip64 )
list(APPEND _cmake_import_check_files_for_hip::amdhip64 "${_IMPORT_PREFIX}/lib/amdhip64_7.lib" "${_IMPORT_PREFIX}/bin/amdhip64_7.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
