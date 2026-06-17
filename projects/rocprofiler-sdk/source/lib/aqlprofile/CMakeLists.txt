#
# Build static AQLProfile library
#

configure_file(${CMAKE_CURRENT_LIST_DIR}/version.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/version.h @ONLY)

if(NOT ROCPROFILER_BUILD_AQLPROFILE)
    rocprofiler_add_interface_library(
        rocprofiler-sdk-aqlprofile "Interface library to external AQLProfile library"
        INTERNAL)
    target_include_directories(rocprofiler-sdk-aqlprofile
                               INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source>)
    target_link_libraries(
        rocprofiler-sdk-aqlprofile
        INTERFACE $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-aqlprofile-external>)
    return()
endif()

rocprofiler_deactivate_clang_tidy()

add_library(rocprofiler-sdk-aqlprofile STATIC)
add_library(rocprofiler-sdk::rocprofiler-sdk-aqlprofile ALIAS rocprofiler-sdk-aqlprofile)
target_include_directories(rocprofiler-sdk-aqlprofile
                           INTERFACE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/source>)
target_link_libraries(
    rocprofiler-sdk-aqlprofile
    PRIVATE $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-headers>
            # $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-build-flags>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-threading>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-memcheck>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-cxx-filesystem>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-abseil>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-fmt>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-dl>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-atomic>
            $<BUILD_INTERFACE:rocprofiler-sdk::rocprofiler-sdk-hsa-runtime-nolink>)
set_target_properties(rocprofiler-sdk-aqlprofile PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(pm4)
