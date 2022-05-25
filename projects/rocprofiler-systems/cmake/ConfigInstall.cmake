# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

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
set(PROJECT_BUILD_TARGETS user dl)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-config.cmake.in
    ${PROJECT_BINARY_DIR}/install-tree/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/omnitrace
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS PROJECT_INSTALL_DIR INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/install-tree/${PROJECT_NAME}-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(
    FILES ${PROJECT_BINARY_DIR}/install-tree/${PROJECT_NAME}-config.cmake
          ${PROJECT_BINARY_DIR}/install-tree/${PROJECT_NAME}-version.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}
    OPTIONAL)

export(PACKAGE ${PROJECT_NAME})
