# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME rocpd)
set(SDK_PACKAGE_NAME "${PROJECT_NAME}")
set(PACKAGE_NAME "${PROJECT_NAME}-rocpd")

install(
    EXPORT ${PACKAGE_NAME}-targets
    FILE ${PACKAGE_NAME}-targets.cmake
    NAMESPACE ${PACKAGE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT rocpd)

rocprofiler_install_env_setup_files(
    NAME ${PACKAGE_NAME}
    VERSION ${PROJECT_VERSION}
    INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}
    COMPONENT rocpd)

# ------------------------------------------------------------------------------#
# install tree
#
set(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(PROJECT_BUILD_TARGETS ${PACKAGE_NAME}-shared-library)
set(PROJECT_EXTRA_DIRS "${CMAKE_INSTALL_INCLUDEDIR}/${PACKAGE_NAME}")

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PACKAGE_NAME}/config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS PROJECT_INSTALL_DIR INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT rocpd)

export(PACKAGE ${PACKAGE_NAME})

# ------------------------------------------------------------------------------#
# build tree
#
set(${PACKAGE_NAME}_BUILD_TREE
    ON
    CACHE BOOL "" FORCE)

set(PROJECT_BUILD_TREE_TARGETS ${SDK_PACKAGE_NAME}::${PACKAGE_NAME}-shared-library
                               ${SDK_PACKAGE_NAME}::${SDK_PACKAGE_NAME}-stack-protector)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PACKAGE_NAME}/build-config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-build-config.cmake
    @ONLY)

file(RELATIVE_PATH rocp_bin2src_rel_path ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})
string(REPLACE "//" "/" rocp_inc_rel_path "${rocp_bin2src_rel_path}/source/include")

set(_BUILDTREE_EXPORT_DIR
    "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${rocp_inc_rel_path}
            ${PROJECT_BINARY_DIR}/include WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

if(NOT EXISTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

if(NOT EXISTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}")
    file(MAKE_DIRECTORY "${_BUILDTREE_EXPORT_DIR}")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}/${PACKAGE_NAME}-targets.cmake")
    file(TOUCH "${_BUILDTREE_EXPORT_DIR}/${PACKAGE_NAME}-targets.cmake")
endif()

export(
    EXPORT ${PACKAGE_NAME}-targets
    NAMESPACE ${PACKAGE_NAME}::
    FILE "${_BUILDTREE_EXPORT_DIR}/${PACKAGE_NAME}-targets.cmake")

set(${PACKAGE_NAME}_DIR
    "${_BUILDTREE_EXPORT_DIR}"
    CACHE PATH "${PACKAGE_NAME} build tree install" FORCE)

install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PACKAGE_NAME}
    COMPONENT rocpd)
