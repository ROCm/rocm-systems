# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-run target
#
# ------------------------------------------------------------------------------#

add_executable(
    rocprofiler-systems-run
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-run.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-run.hpp
    ${CMAKE_CURRENT_LIST_DIR}/impl.cpp
)

target_compile_definitions(rocprofiler-systems-run PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(rocprofiler-systems-run PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    rocprofiler-systems-run
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-common-library
        rocprofiler-systems::rocprofiler-systems-core
        rocprofiler-systems::rocprofiler-systems-sanitizer
)
set_target_properties(
    rocprofiler-systems-run
    PROPERTIES
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-run
)

rocprofiler_systems_strip_target(rocprofiler-systems-run)

install(TARGETS rocprofiler-systems-run DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
