# ======================================================================================
# Builds OTF2
# ======================================================================================

set(ROCPROFILER_BINARY_DIR ${PROJECT_BINARY_DIR})
set(OTF2_VERSION
    "3.0.3"
    CACHE STRING "OTF2 version")
set(OTF2_URL_HASH
    "SHA256=18a3905f7917340387e3edc8e5766f31ab1af41f4ecc5665da6c769ca21c4ee8"
    CACHE STRING "OTF2 URL download hash")

project(
    OTF2
    LANGUAGES C
    VERSION ${OTF2_VERSION}
    DESCRIPTION "Open Trace Format v2"
    HOMEPAGE_URL "https://perftools.pages.jsc.fz-juelich.de/cicd/otf2")

include(FetchContent)
include(ExternalProject)
include(ProcessorCount)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

set(FETCHCONTENT_BASE_DIR ${ROCPROFILER_BINARY_DIR}/external/packages)

fetchcontent_declare(
    otf2-source
    URL https://perftools.pages.jsc.fz-juelich.de/cicd/otf2/tags/otf2-${OTF2_VERSION}/otf2-${OTF2_VERSION}.tar.gz
    URL_HASH ${OTF2_URL_HASH})

fetchcontent_makeavailable(otf2-source)

set(_otf2_root ${ROCPROFILER_BINARY_DIR}/external/otf2)
set(_otf2_inc_dirs $<BUILD_INTERFACE:${_otf2_root}/include>)
set(_otf2_lib_dirs $<BUILD_INTERFACE:${_otf2_root}/${CMAKE_INSTALL_LIBDIR}>)
set(_otf2_libs
    $<BUILD_INTERFACE:${_otf2_root}/${CMAKE_INSTALL_LIBDIR}/libotf2${CMAKE_STATIC_LIBRARY_SUFFIX}>
    )
set(_otf2_build_byproducts
    "${_otf2_root}/include/otf2/otf2.h"
    "${_otf2_root}/${CMAKE_INSTALL_LIBDIR}/libotf2${CMAKE_STATIC_LIBRARY_SUFFIX}")

find_program(
    MAKE_COMMAND
    NAMES make gmake
    PATH_SUFFIXES bin REQUIRED)

externalproject_add(
    otf2-build
    PREFIX ${_otf2_root}
    SOURCE_DIR ${otf2-source_SOURCE_DIR}
    BUILD_IN_SOURCE 1
    DOWNLOAD_COMMAND ""
    PATCH_COMMAND ""
    CONFIGURE_COMMAND
        ${CMAKE_COMMAND} -E env CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
        <SOURCE_DIR>/configure -q --enable-silent-rules --prefix=${_otf2_root}
        --libdir=${_otf2_root}/${CMAKE_INSTALL_LIBDIR} CFLAGS=-fPIC\ -O3\ -g
        CXXFLAGS=-fPIC\ -O3\ -g LDFLAGS= PYTHON=: SPHINX=:
    BUILD_COMMAND ${MAKE_COMMAND} install -s
    BUILD_BYPRODUCTS "${_otf2_build_byproducts}"
    INSTALL_COMMAND "")

add_library(otf2 INTERFACE)
add_library(otf2::otf2 ALIAS otf2)
target_include_directories(otf2 SYSTEM INTERFACE ${_otf2_inc_dirs})
target_link_directories(otf2 INTERFACE ${_otf2_lib_dirs})
target_link_libraries(otf2 INTERFACE ${_otf2_libs})

add_dependencies(otf2 otf2-build)
