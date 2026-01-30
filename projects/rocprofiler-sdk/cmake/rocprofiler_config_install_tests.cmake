# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tests)
set(SDK_PACKAGE_NAME "${PROJECT_NAME}")
set(PACKAGE_NAME "${PROJECT_NAME}-tests")

set(${PACKAGE_NAME}_BUILD_TREE
    ON
    CACHE BOOL "" FORCE)

set(_ROCPROFILER_INSTALL_TEST_COMPONENT OFF)
if(ROCPROFILER_BUILD_TESTS OR ROCPROFILER_THEROCK_PACKAGING)
    set(_ROCPROFILER_INSTALL_TEST_COMPONENT ON)
endif()

# do not install the package config if tests are not requested
if(NOT _ROCPROFILER_INSTALL_TEST_COMPONENT)
    return()
endif()

# ------------------------------------------------------------------------------#
# install tree
#
set(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(TEST_ROOT_DIR ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests)
set(INCLUDE_INSTALL_DIR ${TEST_ROOT_DIR}/unit-tests)
string(REPLACE "-" "_" PACKAGE_NAME_UNDERSCORED ${PACKAGE_NAME})
set(PROJECT_INCLUDE_FILES "")
set(PROJECT_HAS_TEST_TARGETS OFF)
set(PACKAGE_TEST_LD_LIBRARY_PATH
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_INSTALL_PREFIX}/llvm/lib")

if(ROCPROFILER_BUILD_TESTS OR ROCPROFILER_THEROCK_PACKAGING)
    set(PROJECT_HAS_TEST_TARGETS ON)
    install(
        EXPORT ${PACKAGE_NAME}-targets
        FILE ${PACKAGE_NAME}-targets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
        COMPONENT tests)

    get_property(
        _ROCP_SDK_TEST_TARGETS
        DIRECTORY ${CMAKE_SOURCE_DIR}
        PROPERTY rocprofiler-sdk-tests-targets)

    if(_ROCP_SDK_TEST_TARGETS STREQUAL "_ROCP_SDK_TEST_TARGETS-NOTFOUND")
        set(_ROCP_SDK_TEST_TARGETS "")
    endif()

    foreach(_TARG ${_ROCP_SDK_TEST_TARGETS})
        list(APPEND PROJECT_INCLUDE_FILES "${_TARG}.cmake")
    endforeach()
endif()

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PACKAGE_NAME}/config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS PROJECT_INSTALL_DIR INCLUDE_INSTALL_DIR TEST_ROOT_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ExactVersion)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT tests)

rocprofiler_install_env_setup_files(
    NAME ${PACKAGE_NAME}
    VERSION ${PROJECT_VERSION}
    INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}
    COMPONENT tests)

# ------------------------------------------------------------------------------#
# build tree
#
install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE.md
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PACKAGE_NAME}
    COMPONENT tests)
