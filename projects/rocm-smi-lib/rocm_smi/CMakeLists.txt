#
# Minimum version of cmake required
#

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake ROCm SMI (Library)                       ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

## Verbose output.
set(CMAKE_VERBOSE_MAKEFILE on)

# Required Defines first:

message("")
message("Build Configuration:")
# message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
# message("------------Compiler: " ${CMAKE_CXX_COMPILER})
# message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
# message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
# message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
# message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
# message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})
# message("--------RSMI Inc Dir: " ${RSMI_INC_DIR})
# message("")

set(ROCM_SMI "rocm_smi")
set(ROCM_SMI_COMPONENT "lib${ROCM_SMI}")
set(ROCM_SMI_TARGET "${ROCM_SMI}64")

## Include common cmake modules
include(utils)

################# Determine the library version #########################
set(SO_VERSION_GIT_TAG_PREFIX "rsmi_so_ver")

# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

# Debian package specific variables
# Set a default value for the package version
get_version_from_tag("1.0.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)

# VERSION_* variables should be set by get_version_from_tag
if ( ${ROCM_PATCH_VERSION} )
    set ( VERSION_PATCH ${ROCM_PATCH_VERSION})
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
endif ()
set(${ROCM_SMI}_VERSION_MAJOR "${CPACK_PACKAGE_VERSION_MAJOR}")
set(${ROCM_SMI}_VERSION_MINOR "${CPACK_PACKAGE_VERSION_MINOR}")
set(${ROCM_SMI}_VERSION_PATCH "${CPACK_PACKAGE_VERSION_PATCH}")
set(${ROCM_SMI}_VERSION_BUILD "0")
set(${ROCM_SMI}_VERSION_HASH "${PKG_VERSION_HASH}")
message("SOVERSION: ${SO_VERSION_STRING}")

# Configure rsmiBindingsInit.py.in with SO major version:
configure_file(
  "${COMMON_SRC_ROOT}/python_smi_tools/rsmiBindingsInit.py.in"
  "${COMMON_SRC_ROOT}/python_smi_tools/rsmiBindingsInit.py")

# Create a configure file to get version info from within library
configure_file(
  "${PROJECT_SOURCE_DIR}/src/${ROCM_SMI_TARGET}Config.in"
  "${PROJECT_SOURCE_DIR}/include/rocm_smi/${ROCM_SMI_TARGET}Config.h")

set(RSMI_SRC_DIR "src")
set(RSMI_INC_DIR "include")
set(RSMI_DOCS_DIR "docs")

# Add any rocm_smi_lib specific source files here
set(SMI_SRC_LIST ${CMN_SRC_LIST})

# Add any rocm_smi_lib specific headers here
set(SMI_INC_LIST "")

set(SMI_EXAMPLE_EXE "rocm_smi_ex")

add_executable(${SMI_EXAMPLE_EXE} "example/rocm_smi_example.cc")
target_link_libraries(${SMI_EXAMPLE_EXE} ${ROCM_SMI_TARGET})
add_library(${ROCM_SMI_TARGET}  ${CMN_SRC_LIST} ${SMI_SRC_LIST}
                                              ${CMN_INC_LIST} ${SMI_INC_LIST})
target_link_libraries(${ROCM_SMI_TARGET} PRIVATE pthread rt dl)
target_include_directories(${ROCM_SMI_TARGET} PRIVATE
          ${CMAKE_CURRENT_SOURCE_DIR} ${COMMON_PROJ_ROOT}/common/shared_mutex)

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${ROCM_SMI_TARGET}
                           PUBLIC
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

if(FILE_REORG_BACKWARD_COMPATIBILITY)
    target_include_directories(${ROCM_SMI_TARGET}
                           PUBLIC
                           "$<INSTALL_INTERFACE:${ROCM_SMI}/include>")
endif()

## Set the VERSION and SOVERSION values
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY
             SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY
             VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if ("${CMAKE_BUILD_TYPE}" STREQUAL Release)
  if(${BUILD_SHARED_LIBS}) #stripping only for .so
    add_custom_command(
          TARGET ${ROCM_SMI_TARGET}
          POST_BUILD COMMAND ${CMAKE_STRIP} lib${ROCM_SMI_TARGET}.so.${SO_VERSION_STRING})
  endif()
endif ()

#file reorganization changes
#rocm_smi.py moved to libexec/rocm_smi. so creating rocm-smi symlink
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
add_custom_target(link-rocm-smi ALL
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  COMMAND ${CMAKE_COMMAND} -E create_symlink
                  ../${CMAKE_INSTALL_LIBEXECDIR}/${ROCM_SMI}/rocm_smi.py ${CMAKE_CURRENT_BINARY_DIR}/bin/rocm-smi)

## Add the install directives for the runtime library.
install(TARGETS ${ROCM_SMI_TARGET}
          EXPORT rocm_smiTargets
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev)
install(TARGETS ${ROCM_SMI_TARGET}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          COMPONENT asan)

install(FILES ${COMMON_SRC_ROOT}/include/rocm_smi/rocm_smi.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm_smi
        COMPONENT dev)
install(FILES ${COMMON_SRC_ROOT}/include/rocm_smi/${ROCM_SMI_TARGET}Config.h
       DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm_smi
       COMPONENT dev)
install(FILES ${COMMON_SRC_ROOT}/include/rocm_smi/kfd_ioctl.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm_smi
        COMPONENT dev)
install(PROGRAMS ${COMMON_SRC_ROOT}/python_smi_tools/rsmiBindingsInit.py
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${ROCM_SMI}
        COMPONENT dev)
install(PROGRAMS ${COMMON_SRC_ROOT}/python_smi_tools/rsmiBindings.py
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${ROCM_SMI}
        COMPONENT dev)
install(PROGRAMS ${COMMON_SRC_ROOT}/python_smi_tools/rocm_smi.py
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${ROCM_SMI}
        COMPONENT dev)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bin/rocm-smi
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT dev)
