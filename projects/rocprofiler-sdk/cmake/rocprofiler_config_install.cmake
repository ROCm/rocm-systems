# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME development)

install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PACKAGE_NAME}
    COMPONENT core)

if(ROCPROFILER_BUILD_DOCS)
    install(
        FILES ${PROJECT_SOURCE_DIR}/LICENSE
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PACKAGE_NAME}-docs
        COMPONENT docs)
endif()

install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${PACKAGE_NAME}-tests
    COMPONENT tests)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/samples
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}
    COMPONENT samples)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}
    COMPONENT tests)

install(
    FILES ${PROJECT_SOURCE_DIR}/requirements.txt
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PACKAGE_NAME}/tests
    COMPONENT tests)

install(
    EXPORT ${PACKAGE_NAME}-targets
    FILE ${PACKAGE_NAME}-targets.cmake
    NAMESPACE ${PACKAGE_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT development)

install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/Modules/rocprofiler-sdk-custom-compilation.cmake
          ${PROJECT_SOURCE_DIR}/cmake/Modules/rocprofiler-sdk-utilities.cmake
          ${PROJECT_SOURCE_DIR}/cmake/Modules/Findlibdw.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/Modules
    COMPONENT development)

rocprofiler_install_env_setup_files(
    NAME ${PACKAGE_NAME}
    VERSION ${PROJECT_VERSION}
    INSTALL_DIR ${CMAKE_INSTALL_DATAROOTDIR}
    COMPONENT development)

function(compute_rocprofiler_sdk_version _VAR)
    string(REGEX REPLACE "([0-9]+)\\\.([0-9]+)\\\.(.*)" "\\1.\\2" _TMP "${${_VAR}}")
    set(PACKAGE_${_VAR}
        "${_TMP}.0...${_TMP}.999999999999"
        PARENT_SCOPE)
endfunction()

compute_rocprofiler_sdk_version(amd_comgr_VERSION)
compute_rocprofiler_sdk_version(hsa-runtime64_VERSION)
compute_rocprofiler_sdk_version(hip_VERSION)

# ------------------------------------------------------------------------------#
# install tree
#
set(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(PROJECT_BUILD_TARGETS headers shared-library)
set(PROJECT_EXTRA_DIRS "${CMAKE_INSTALL_INCLUDEDIR}/${PACKAGE_NAME}"
                       "${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}")

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

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/rocprofiler_config_nolink_target.cmake
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-nolink-target.cmake
    COPYONLY)

foreach(_FILE rocprofiler-sdk-custom-compilation.cmake rocprofiler-sdk-utilities.cmake
              Findlibdw.cmake)
    configure_file(
        ${PROJECT_SOURCE_DIR}/cmake/Modules/${_FILE}
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/Modules/${_FILE}
        COPYONLY)
endforeach()

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-version.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}/${PACKAGE_NAME}-config-nolink-target.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PACKAGE_NAME}
    COMPONENT development)

export(PACKAGE ${PACKAGE_NAME})

# ------------------------------------------------------------------------------#
# build tree
#
set(${PACKAGE_NAME}_BUILD_TREE
    ON
    CACHE BOOL "" FORCE)

set(PROJECT_BUILD_TREE_TARGETS headers shared-library build-flags stack-protector dw)

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
