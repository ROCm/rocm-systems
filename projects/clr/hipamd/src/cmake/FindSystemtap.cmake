# FindSystemtap.cmake
#
# Locates <sys/sdt.h> from systemtap-sdt-dev / systemtap-sdt-devel.
# Defines:
#   Systemtap_FOUND
#   Systemtap_INCLUDE_DIR
#   Systemtap::sdt (INTERFACE imported target)

find_path(Systemtap_INCLUDE_DIR
    NAMES sys/sdt.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES ${CMAKE_LIBRARY_ARCHITECTURE} x86_64-linux-gnu aarch64-linux-gnu
    DOC "Path to <sys/sdt.h> from systemtap-sdt-dev")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Systemtap
    REQUIRED_VARS Systemtap_INCLUDE_DIR)

if(Systemtap_FOUND AND NOT TARGET Systemtap::sdt)
    add_library(Systemtap::sdt INTERFACE IMPORTED)
    set_target_properties(Systemtap::sdt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${Systemtap_INCLUDE_DIR}")
endif()

mark_as_advanced(Systemtap_INCLUDE_DIR)
