# ------------------------------------------------------------------------------#
#
# rocprofiler-systems common headers
#
# ------------------------------------------------------------------------------#

add_library(rocprofiler-systems-common-library INTERFACE)
add_library(rocprofiler-systems::common-library ALIAS rocprofiler-systems-common-library)
add_library(
    rocprofiler-systems::rocprofiler-systems-common-library
    ALIAS rocprofiler-systems-common-library
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/defines.h
    @ONLY
)

target_sources(
    rocprofiler-systems-common-library
    INTERFACE
        ${CMAKE_CURRENT_BINARY_DIR}/defines.h
        ${CMAKE_CURRENT_SOURCE_DIR}/delimit.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/environment.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/invoke.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/join.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/setup.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/traits.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/md5sum.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/static_object.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/synchronized.hpp
)

get_filename_component(COMMON_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(COMMON_BINARY_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}" DIRECTORY)

target_include_directories(
    rocprofiler-systems-common-library
    INTERFACE
        $<BUILD_INTERFACE:${COMMON_SOURCE_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${COMMON_BINARY_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/timemory/source>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/timemory/source>
)

target_compile_definitions(
    rocprofiler-systems-common-library
    INTERFACE $<BUILD_INTERFACE:ROCPROFSYS_INTERNAL_BUILD=1>
)
