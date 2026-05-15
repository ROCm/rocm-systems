# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME tests)
set(SDK_PACKAGE_NAME "${PROJECT_NAME}")
set(PACKAGE_NAME "${PROJECT_NAME}-tests")

set(${PACKAGE_NAME}_BUILD_TREE
    ON
    CACHE BOOL "" FORCE)

# do not install the package config if tests are not built
if(NOT ROCPROFILER_BUILD_TESTS)
    return()
endif()

install(
    EXPORT ${PACKAGE_NAME}-targets
    FILE ${PACKAGE_NAME}-targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT tests)

rocprofiler_install_env_setup_files(
    NAME ${PACKAGE_NAME}
    VERSION ${PROJECT_VERSION}
    INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}
    COMPONENT tests)

# ------------------------------------------------------------------------------#
# install tree
#
set(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/unit-tests)
string(REPLACE "-" "_" PACKAGE_NAME_UNDERSCORED ${PACKAGE_NAME})
get_property(
    _ROCP_SDK_TEST_TARGETS
    DIRECTORY ${CMAKE_SOURCE_DIR}
    PROPERTY rocprofiler-sdk-tests-targets)

foreach(_TARG ${_ROCP_SDK_TEST_TARGETS})
    list(APPEND PROJECT_INCLUDE_FILES "${_TARG}.cmake")
endforeach()

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PACKAGE_NAME}/config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS PROJECT_INSTALL_DIR INCLUDE_INSTALL_DIR)

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

# ------------------------------------------------------------------------------#
# build tree
#
install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE.md
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PACKAGE_NAME}
    COMPONENT tests)
