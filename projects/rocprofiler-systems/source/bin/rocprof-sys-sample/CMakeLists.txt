# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-sample target
#
# ------------------------------------------------------------------------------#

add_executable(
    rocprofiler-systems-sample
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-sample.cpp
    ${CMAKE_CURRENT_LIST_DIR}/impl.cpp
)

target_compile_definitions(rocprofiler-systems-sample PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(rocprofiler-systems-sample PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    rocprofiler-systems-sample
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-common-library
)
set_target_properties(
    rocprofiler-systems-sample
    PROPERTIES
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-sample
)

rocprofiler_systems_strip_target(rocprofiler-systems-sample)

install(TARGETS rocprofiler-systems-sample DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
