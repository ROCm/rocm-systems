# include guard
include_guard(GLOBAL)

include(CMakePackageConfigHelpers)

install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT core)

install(
    EXPORT rocprofiler-register-library-targets
    FILE rocprofiler-register-library-targets.cmake
    NAMESPACE rocprofiler-register::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    COMPONENT core)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/tests
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    COMPONENT tests)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/setup-env.sh.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/setup-env.sh @ONLY)

configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/modulefile.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${PROJECT_VERSION}
    @ONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/setup-env.sh
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}
    COMPONENT core)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}/${PROJECT_VERSION}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/modulefiles/${PROJECT_NAME}
    COMPONENT core)

# ------------------------------------------------------------------------------#
# install tree
#
set(PROJECT_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/Templates/${PROJECT_NAME}-config.cmake.in
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocprofiler-register
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    PATH_VARS PROJECT_INSTALL_DIR INCLUDE_INSTALL_DIR LIB_INSTALL_DIR)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-config.cmake
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}/${PROJECT_NAME}-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    OPTIONAL)

# ------------------------------------------------------------------------------#
# build tree
#
set(${PROJECT_NAME}_BUILD_TREE
    ON
    CACHE BOOL "" FORCE)

file(RELATIVE_PATH rocp_reg_bin2src_rel_path ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR})
string(REPLACE "//" "/" rocp_reg_inc_rel_path
               "${rocp_reg_bin2src_rel_path}/source/include")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${rocp_reg_inc_rel_path}
            ${PROJECT_BINARY_DIR}/include WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(_BUILDTREE_EXPORT_DIR
    "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/rocprofiler-register")

if(NOT EXISTS "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}")
    file(MAKE_DIRECTORY "${_BUILDTREE_EXPORT_DIR}")
endif()

if(NOT EXISTS "${_BUILDTREE_EXPORT_DIR}/rocprofiler-register-library-targets.cmake")
    file(TOUCH "${_BUILDTREE_EXPORT_DIR}/rocprofiler-register-library-targets.cmake")
endif()

export(
    EXPORT rocprofiler-register-library-targets
    NAMESPACE rocprofiler-register::
    FILE "${_BUILDTREE_EXPORT_DIR}/rocprofiler-register-library-targets.cmake")

set(rocprofiler-register_DIR
    "${_BUILDTREE_EXPORT_DIR}"
    CACHE PATH "rocprofiler-register" FORCE)
