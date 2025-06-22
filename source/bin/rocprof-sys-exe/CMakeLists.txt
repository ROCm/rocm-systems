# ------------------------------------------------------------------------------#
#
# TODO: [DFG] Remove this file after 'rocprofsys' rebranding is complete
# rocprofiler-systems-exe target (deprecated 'omnitrace' executable, now 'rocprofiler-systems-instrument')
#
# ------------------------------------------------------------------------------#

add_executable(rocprofiler-systems-exe ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys.cpp)

target_link_libraries(
    rocprofiler-systems-exe
    PRIVATE rocprofiler-systems::rocprofiler-systems-threading
)

set_target_properties(
    rocprofiler-systems-exe
    PROPERTIES
        OUTPUT_NAME rocprof-sys
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH ON
)

rocprofiler_systems_strip_target(rocprofiler-systems-exe)

install(TARGETS rocprofiler-systems-exe DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
