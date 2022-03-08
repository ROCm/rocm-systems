# ------------------------------------------------------------------------------#
#
# omnitrace common headers
#
# ------------------------------------------------------------------------------#

add_library(omnitrace-common-library INTERFACE)
add_library(omnitrace::common-library ALIAS omnitrace-common-library)

target_sources(
    omnitrace-common-library
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/defines.h
              ${CMAKE_CURRENT_SOURCE_DIR}/delimit.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/environment.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/invoke.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/join.hpp)

get_filename_component(COMMON_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)

target_include_directories(omnitrace-common-library INTERFACE ${COMMON_INCLUDE_DIR})
target_compile_definitions(omnitrace-common-library
                           INTERFACE $<BUILD_INTERFACE:OMNITRACE_INTERNAL_BUILD>)
