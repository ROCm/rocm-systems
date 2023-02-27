# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME config)

install(
    EXPORT omnitrace-library-targets
    FILE omnitrace-library-targets.cmake
    NAMESPACE omnitrace::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/omnitrace)

# ------------------------------------------------------------------------------#
# install tree
#
set(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(PROJECT_BUILD_TARGETS user)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/omnitrace
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS PROJECT_INSTALL_DIR INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-config.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-version.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    OPTIONAL)

export(PACKAGE ${PROJECT_NAME})

# ------------------------------------------------------------------------------#
# install the validate-causal-json python script as a utility
#
configure_file(
    ${PROJECT_SOURCE_DIR}/tests/validate-causal-json.py
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/omnitrace-causal-print COPYONLY)

install(PROGRAMS ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/omnitrace-causal-print
        DESTINATION ${CMAKE_INSTALL_BINDIR})

# ------------------------------------------------------------------------------#
# build tree
#
set(_BUILDTREE_EXPORT_DIR
    "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/omnitrace")

if(NOT EXISTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}")
    file(MAKE_DIRECTORY "${_BUILDTREE_EXPORT_DIR}")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}/omnitrace-library-targets.cmake")
    file(TOUCH "${_BUILDTREE_EXPORT_DIR}/omnitrace-library-targets.cmake")
endif()

export(
    EXPORT omnitrace-library-targets
    NAMESPACE omnitrace::
    FILE "${_BUILDTREE_EXPORT_DIR}/omnitrace-library-targets.cmake")

set(omnitrace_DIR
    "${_BUILDTREE_EXPORT_DIR}"
    CACHE PATH "omnitrace" FORCE)
