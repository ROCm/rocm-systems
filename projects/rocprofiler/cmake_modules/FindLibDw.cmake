# Try to find LIBDW
#
# Once found, this will define:
#   - LIBDW_FOUND - system has libelf
#   - LIBDW_INCLUDE_DIRS - the libelf include directory
#   - LIBDW_LIBRARIES - Link these to use libelf
#   - LIBDW_DEFINITIONS - Compiler switches required for using libelf
find_path(
    FIND_LIBDW_INCLUDES
    NAMES elfutils/libdw.h
    PATHS /usr/include /usr/local/include)

find_library(FIND_LIBDW_LIBRARIES NAMES dw PATH /usr/lib /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibDw DEFAULT_MSG FIND_LIBDW_INCLUDES
                                  FIND_LIBDW_LIBRARIES)
mark_as_advanced(FIND_LIBDW_INCLUDES FIND_LIBDW_LIBRARIES)

set(LIBDW_INCLUDES ${FIND_LIBDW_INCLUDES})
set(LIBDW_LIBRARIES ${FIND_LIBDW_LIBRARIES})
