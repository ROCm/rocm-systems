#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "wkmi::wkmi" for configuration "Release"
set_property(TARGET wkmi::wkmi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(wkmi::wkmi PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib64/libwkmi.a"
  )

list(APPEND _cmake_import_check_targets wkmi::wkmi )
list(APPEND _cmake_import_check_files_for_wkmi::wkmi "${_IMPORT_PREFIX}/lib64/libwkmi.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
