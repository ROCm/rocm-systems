# executable RPATH

set(ROCPROFSYS_EXE_INSTALL_RPATH
    "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}"
)

# common library (shared by run, sample, etc.)
add_subdirectory(common)

# executables
add_subdirectory(rocprof-sys-avail)
add_subdirectory(rocprof-sys-causal)
add_subdirectory(rocprof-sys-sample)
add_subdirectory(rocprof-sys-instrument)
add_subdirectory(rocprof-sys-run)
add_subdirectory(rocprof-sys-attach)

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(common/tests)
endif()
