# ------------------------------------------------------------------------------#
#
# rocprofiler-systems dl library
#
# ------------------------------------------------------------------------------#

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_SKIP_RPATH OFF)
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
set(CMAKE_CXX_VISIBILITY_PRESET "internal")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

add_library(rocprofiler-systems-dl-library SHARED)
add_library(
    rocprofiler-systems::rocprofiler-systems-dl-library
    ALIAS rocprofiler-systems-dl-library
)

target_sources(
    rocprofiler-systems-dl-library
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/dl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/main.c
        ${CMAKE_CURRENT_SOURCE_DIR}/dl/dl.hpp
)
target_include_directories(
    rocprofiler-systems-dl-library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../rocprof-sys-user>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../rocprof-sys>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${rocprofiler-sdk_INCLUDE_DIR}
)
target_link_libraries(
    rocprofiler-systems-dl-library
    PUBLIC
        $<BUILD_INTERFACE:${dl_LIBRARY}>
        $<BUILD_INTERFACE:rocprofiler-systems::common-library>
        $<BUILD_INTERFACE:rocprofiler-systems::rocprofiler-systems-compile-definitions>
        $<BUILD_INTERFACE:Threads::Threads>
)

add_target_cxx_flag_if_avail(rocprofiler-systems-dl-library "-ftls-model=global-dynamic")
add_target_cxx_flag_if_avail(rocprofiler-systems-dl-library "-g3")

set_target_properties(
    rocprofiler-systems-dl-library
    PROPERTIES
        OUTPUT_NAME ${BINARY_NAME_PREFIX}-dl
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN"
)

rocprofiler_systems_strip_target(rocprofiler-systems-dl-library)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/dl/dl.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/dl
)

install(TARGETS rocprofiler-systems-dl-library DESTINATION ${CMAKE_INSTALL_LIBDIR})
