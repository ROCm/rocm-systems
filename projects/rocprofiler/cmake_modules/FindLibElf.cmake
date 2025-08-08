# Try to find LIBELF
#
# Once found, this will define:
#   - LIBELF_FOUND - system has libelf
#   - LIBELF_INCLUDE_DIRS - the libelf include directory
#   - LIBELF_LIBRARIES - Link these to use libelf
#   - LIBELF_DEFINITIONS - Compiler switches required for using libelf
find_path(
    FIND_LIBELF_INCLUDES
    NAMES libelf.h
    PATHS /usr/include /usr/include/libelf /usr/local/include /usr/local/include/libelf)

find_library(FIND_LIBELF_LIBRARIES NAMES elf PATH /usr/lib /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibElf DEFAULT_MSG FIND_LIBELF_INCLUDES
                                  FIND_LIBELF_LIBRARIES)
mark_as_advanced(FIND_LIBELF_INCLUDES FIND_LIBELF_LIBRARIES)

set(LIBELF_INCLUDES ${FIND_LIBELF_INCLUDES})
set(LIBELF_LIBRARIES ${FIND_LIBELF_LIBRARIES})
