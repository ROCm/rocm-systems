# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-causal target
#
# ------------------------------------------------------------------------------#

add_executable(
    rocprofiler-systems-causal
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-causal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-causal.hpp
    ${CMAKE_CURRENT_LIST_DIR}/impl.cpp
)

target_compile_definitions(rocprofiler-systems-causal PRIVATE TIMEMORY_CMAKE=1)
target_include_directories(rocprofiler-systems-causal PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
    rocprofiler-systems-causal
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-common-library
        rocprofiler-systems::rocprofiler-systems-core
)
set_target_properties(
    rocprofiler-systems-causal
    PROPERTIES
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-causal
)

rocprofiler_systems_strip_target(rocprofiler-systems-causal)

install(TARGETS rocprofiler-systems-causal DESTINATION ${CMAKE_INSTALL_BINDIR} OPTIONAL)
