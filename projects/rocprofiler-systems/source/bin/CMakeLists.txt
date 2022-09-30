# executable RPATH

if(OMNITRACE_USE_ROCPROFILER
   AND rocprofiler_LIBRARY_DIR
   AND ROCmVersion_TRIPLE_VERSION VERSION_LESS 5.2.0
   AND NOT CMAKE_INSTALL_RPATH_USE_LINK_PATH)
    set(OMNITRACE_EXE_INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/omnitrace:${rocprofiler_LIBRARY_DIR}"
        )
else()
    set(OMNITRACE_EXE_INSTALL_RPATH
        "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/omnitrace"
        )
endif()

# executables
add_subdirectory(omnitrace-avail)
add_subdirectory(omnitrace-critical-trace)
add_subdirectory(omnitrace-sample)
add_subdirectory(omnitrace)

if(OMNITRACE_BUILD_TESTING OR "$ENV{OMNITRACE_CI}" MATCHES "[1-9]+|ON|on|y|yes")
    add_subdirectory(tests)
endif()
