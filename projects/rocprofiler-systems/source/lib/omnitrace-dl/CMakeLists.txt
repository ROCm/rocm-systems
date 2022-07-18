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
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../omnitrace-user>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../omnitrace>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
    omnitrace-dl-library
    PUBLIC
        ${dl_LIBRARY}
        $<BUILD_INTERFACE:omnitrace::common-library>
        $<BUILD_INTERFACE:omnitrace::omnitrace-compile-definitions>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_STATIC_LIBGCC}>,omnitrace::omnitrace-static-libgcc,>>
        $<BUILD_INTERFACE:$<IF:$<BOOL:${OMNITRACE_BUILD_STATIC_LIBSTDCXX}>,omnitrace::omnitrace-static-libstdcxx,>>
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

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/dl.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/omnitrace)

install(
    TARGETS omnitrace-dl-library
    EXPORT omnitrace-library-targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR})
