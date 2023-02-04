# ----------------------------------------------------------------------------- #
#
# omnitrace: contains all instrumentation functionality
#
# omnitrace-dl: contains minimal symbols and dlopen's omnitrace
#
# omnitrace-user: contains symbols for user API
#
# ----------------------------------------------------------------------------- #

if(OMNITRACE_USE_ROCPROFILER
   AND rocprofiler_LIBRARY_DIR
   AND ROCmVersion_TRIPLE_VERSION VERSION_LESS 5.2.0
   AND NOT CMAKE_INSTALL_RPATH_USE_LINK_PATH)
    set(OMNITRACE_LIB_INSTALL_RPATH
        "\$ORIGIN:\$ORIGIN/omnitrace:${rocprofiler_LIBRARY_DIR}")
else()
    set(OMNITRACE_LIB_INSTALL_RPATH "\$ORIGIN:\$ORIGIN/omnitrace")
endif()

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(binary)

add_subdirectory(omnitrace)
add_subdirectory(omnitrace-dl)
add_subdirectory(omnitrace-user)
