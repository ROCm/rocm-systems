# ------------------------------------------------------------------------------#
#
# rocprofiler-systems-instrument target (formerly rocprofiler-systems-exe target prior to 1.8.1)
#
# ------------------------------------------------------------------------------#

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_executable(rocprofiler-systems-instrument)

target_sources(
    rocprofiler-systems-instrument
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/details.cpp
        ${CMAKE_CURRENT_LIST_DIR}/function_signature.cpp
        ${CMAKE_CURRENT_LIST_DIR}/function_signature.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
        ${CMAKE_CURRENT_LIST_DIR}/info.hpp
        ${CMAKE_CURRENT_LIST_DIR}/internal_libs.cpp
        ${CMAKE_CURRENT_LIST_DIR}/internal_libs.hpp
        ${CMAKE_CURRENT_LIST_DIR}/log.cpp
        ${CMAKE_CURRENT_LIST_DIR}/log.hpp
        ${CMAKE_CURRENT_LIST_DIR}/module_function.cpp
        ${CMAKE_CURRENT_LIST_DIR}/module_function.hpp
        ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-instrument.cpp
        ${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-instrument.hpp
)

target_link_libraries(
    rocprofiler-systems-instrument
    PRIVATE
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-dyninst
        rocprofiler-systems::rocprofiler-systems-compile-options
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-sanitizer
        timemory::timemory-headers
        timemory::timemory-extensions
        timemory::timemory-core
)

add_target_flag_if_avail(rocprofiler-systems-instrument "-Wno-deprecated-declarations")

set_target_properties(
    rocprofiler-systems-instrument
    PROPERTIES
        BUILD_RPATH "\$ORIGIN:\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        INSTALL_RPATH "${ROCPROFSYS_EXE_INSTALL_RPATH}"
        INSTALL_RPATH_USE_LINK_PATH ON
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-instrument
)

if(ROCPROFSYS_BUILD_DYNINST)
    target_compile_definitions(
        rocprofiler-systems-instrument
        PRIVATE ROCPROFSYS_BUILD_DYNINST=1
    )
endif()

add_target_flag_if_avail(rocprofiler-systems-instrument "-Wno-deprecated-declarations")

rocprofiler_systems_strip_target(rocprofiler-systems-instrument)

if(CMAKE_BUILD_TYPE MATCHES "^(DEBUG|Debug)")
    string(REPLACE " " ";" _FLAGS "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    target_compile_options(rocprofiler-systems-instrument PRIVATE ${_FLAGS})
endif()

install(
    TARGETS rocprofiler-systems-instrument
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    OPTIONAL
)
