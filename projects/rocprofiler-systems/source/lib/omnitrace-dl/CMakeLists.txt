# ------------------------------------------------------------------------------#
#
# omnitrace dl library
#
# ------------------------------------------------------------------------------#

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_SKIP_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "internal")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(omnitrace-dl-library SHARED)
add_library(omnitrace::omnitrace-dl-library ALIAS omnitrace-dl-library)

target_sources(omnitrace-dl-library PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/dl.cpp
                                            ${CMAKE_CURRENT_SOURCE_DIR}/dl.hpp)
target_include_directories(
    omnitrace-dl-library
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../omnitrace-user>)
target_link_libraries(
    omnitrace-dl-library
    PUBLIC
        ${dl_LIBRARY}
        $<BUILD_INTERFACE:omnitrace::common-library>
        $<BUILD_INTERFACE:omnitrace::omnitrace-compile-definitions>
        $<IF:$<BOOL:${OMNITRACE_BUILD_STATIC_LIBGCC}>,$<BUILD_INTERFACE:omnitrace::omnitrace-static-libgcc>,>
        $<IF:$<BOOL:${OMNITRACE_BUILD_STATIC_LIBSTDCXX}>,$<BUILD_INTERFACE:omnitrace::omnitrace-static-libstdcxx>,>
    )

add_target_cxx_flag_if_avail(omnitrace-dl-library "-ftls-model=global-dynamic")
add_target_cxx_flag_if_avail(omnitrace-dl-library "-g")

set_target_properties(
    omnitrace-dl-library
    PROPERTIES OUTPUT_NAME omnitrace-dl
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN")

omnitrace_strip_target(omnitrace-dl-library)

install(
    TARGETS omnitrace-dl-library
    EXPORT omnitrace-dl-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    OPTIONAL)

install(
    EXPORT omnitrace-dl-library-targets
    FILE omnitrace-dl-library-targets.cmake
    NAMESPACE omnitrace::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/omnitrace)
