# executable RPATH
set(OMNITRACE_EXE_INSTALL_RPATH
    "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/omnitrace:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/timemory/libunwind:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/dyninst-tpls/lib"
    )

# executables
add_subdirectory(omnitrace-avail)
add_subdirectory(omnitrace-critical-trace)
add_subdirectory(omnitrace)

if(OMNITRACE_BUILD_TESTING OR "$ENV{OMNITRACE_CI}" MATCHES "[1-9]+|ON|on|y|yes")
    add_subdirectory(tests)
endif()
