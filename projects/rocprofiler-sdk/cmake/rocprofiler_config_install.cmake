# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME config)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/samples
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}-sdk
    COMPONENT samples)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}-sdk
    COMPONENT tests)

install(
    EXPORT rocprofiler-sdk-library-targets
    FILE rocprofiler-sdk-library-targets.cmake
    NAMESPACE rocprofiler::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk
    COMPONENT development)

# ------------------------------------------------------------------------------#
# install tree
#
set(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(PROJECT_BUILD_TARGETS headers shared-library)

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-sdk-config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk/${PROJECT_NAME}-sdk-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/rocprofiler-sdk
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS PROJECT_INSTALL_DIR INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk/${PROJECT_NAME}-sdk-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/rocprofiler_config_nolink_target.cmake
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk/${PROJECT_NAME}-sdk-config-nolink-target.cmake
    COPYONLY)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk/${PROJECT_NAME}-sdk-config.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk/${PROJECT_NAME}-sdk-config-version.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk/${PROJECT_NAME}-sdk-config-nolink-target.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk
    COMPONENT development)

export(PACKAGE ${PROJECT_NAME})

# ------------------------------------------------------------------------------#
# build tree
#
set(${PROJECT_NAME}_BUILD_TREE
    ON
    CACHE BOOL "" FORCE)

set(PROJECT_BUILD_TREE_TARGETS headers shared-library build-flags stack-protector)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-sdk-build-config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk/${PROJECT_NAME}-sdk-build-config.cmake
    @ONLY)

file(RELATIVE_PATH rocp_bin2src_rel_path ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})
string(REPLACE "//" "/" rocp_inc_rel_path "${rocp_bin2src_rel_path}/source/include")

set(_BUILDTREE_EXPORT_DIR
    "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/cmake/${PROJECT_NAME}-sdk")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${rocp_inc_rel_path}
            ${PROJECT_BINARY_DIR}/include WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

if(NOT EXISTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

if(NOT EXISTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}-sdk")
    file(MAKE_DIRECTORY
         "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}-sdk")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}")
    file(MAKE_DIRECTORY "${_BUILDTREE_EXPORT_DIR}")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}/${PROJECT_NAME}-sdk-library-targets.cmake")
    file(TOUCH "${_BUILDTREE_EXPORT_DIR}/${PROJECT_NAME}-sdk-library-targets.cmake")
endif()

export(
    EXPORT rocprofiler-sdk-library-targets
    NAMESPACE rocprofiler::
    FILE "${_BUILDTREE_EXPORT_DIR}/${PROJECT_NAME}-sdk-library-targets.cmake")

set(rocprofiler-sdk_DIR
    "${_BUILDTREE_EXPORT_DIR}"
    CACHE PATH "rocprofiler" FORCE)
