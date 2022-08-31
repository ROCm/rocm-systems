# ------------------------------------------------------------------------------#
#
# omnitrace common headers
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-common-library INTERFACE)
add_library(omnitrace::common-library ALIAS omnitrace-common-library)
add_library(omnitrace::omnitrace-common-library ALIAS omnitrace-common-library)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/defines.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/defines.h @ONLY)

target_sources(
    omnitrace-common-library
    INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/defines.h
              ${CMAKE_CURRENT_SOURCE_DIR}/delimit.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/environment.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/invoke.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/join.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/setup.hpp)

get_filename_component(COMMON_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
get_filename_component(COMMON_BINARY_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}" DIRECTORY)

target_include_directories(
    omnitrace-common-library
    INTERFACE $<BUILD_INTERFACE:${COMMON_SOURCE_INCLUDE_DIR}>
              $<BUILD_INTERFACE:${COMMON_BINARY_INCLUDE_DIR}>
              $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/timemory/source>
              $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/timemory/source>)

target_compile_definitions(omnitrace-common-library
                           INTERFACE $<BUILD_INTERFACE:OMNITRACE_INTERNAL_BUILD=1>)
