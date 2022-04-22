# ----------------------------------------------------------------------------- #
#
# omnitrace: contains all instrumentation functionality
#
# omnitrace-dl: contains minimal symbols and dlopen's omnitrace
#
# omnitrace-user: contains symbols for user API
#
# ----------------------------------------------------------------------------- #

string(REPLACE ":" ";" _INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
if(_INSTALL_RPATH)
    list(REMOVE_DUPLICATES _INSTALL_RPATH)
endif()
string(REPLACE ";" ":" _INSTALL_RPATH "${_INSTALL_RPATH}")
string(
    REPLACE
        "::"
        ":"
        OMNITRACE_LIB_INSTALL_RPATH
        "\$ORIGIN:\$ORIGIN/omnitrace:\$ORIGIN/timemory/libunwind:\$ORIGIN/dyninst-tpls/libs:${_INSTALL_RPATH}"
    )

add_subdirectory(common)
add_subdirectory(omnitrace)
add_subdirectory(omnitrace-dl)
add_subdirectory(omnitrace-user)
