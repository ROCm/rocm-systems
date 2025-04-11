#
# Minimum version of cmake required
#

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    CMake ROCm SMI (Library)                       ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
# message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
# message("------------Compiler: " ${CMAKE_CXX_COMPILER})
# message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
# message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
# message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
# message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
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
if(${ROCM_PATCH_VERSION})
    set(VERSION_PATCH ${ROCM_PATCH_VERSION})
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

# Create a configure file to get version info from within library
configure_file("src/${ROCM_SMI_TARGET}Config.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/include/rocm_smi/${ROCM_SMI_TARGET}Config.h")

set(rocm_smi_VERSION_MAJOR "${VERSION_MAJOR}")
set(rocm_smi_VERSION_MINOR "${VERSION_MINOR}")
set(rocm_smi_VERSION_PATCH "0")
set(rocm_smi_VERSION_BUILD "0")

add_library(${ROCM_SMI_TARGET} ${CMN_SRC_LIST} ${CMN_INC_LIST})
target_link_libraries(${ROCM_SMI_TARGET} pthread rt dl)
target_include_directories(${ROCM_SMI_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                      ${PROJECT_SOURCE_DIR}/common/shared_mutex)

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${ROCM_SMI_TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                     "$<INSTALL_INTERFACE:${ROCM_SMI}/include>")

## Set the VERSION and SOVERSION values
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${ROCM_SMI_TARGET} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(${BUILD_SHARED_LIBS}) #stripping only for .so
        add_custom_command(
            TARGET ${ROCM_SMI_TARGET}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} lib${ROCM_SMI_TARGET}.so.${SO_VERSION_STRING})
    endif()
endif()

#file reorganization changes
#rocm_smi.py moved to libexec/rocm_smi. so creating rocm-smi symlink
#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
#add_custom_target(
#    link-rocm-smi ALL
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    COMMAND ${CMAKE_COMMAND} -E create_symlink ../libexec/${ROCM_SMI}/rocm_smi.py
#            ${CMAKE_CURRENT_BINARY_DIR}/bin/rocm-smi)

## Add the install directives for the runtime library.
#install(
#    TARGETS ${ROCM_SMI_TARGET}
#    EXPORT rocm_smiTargets
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${ROCM_SMI_COMPONENT})
#
#install(
#    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rocm_smi/rocm_smi.h
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm_smi
#    COMPONENT ${ROCM_SMI_COMPONENT})
#install(
#    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rocm_smi/${ROCM_SMI_TARGET}Config.h
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm_smi
#    COMPONENT ${ROCM_SMI_COMPONENT})
#install(
#    FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/rocm_smi/kfd_ioctl.h
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm_smi
#    COMPONENT ${ROCM_SMI_COMPONENT})
#install(FILES ${PROJECT_SOURCE_DIR}/rocm_smi/python_smi_tools/rsmiBindings.py
#                                        DESTINATION libexec/${ROCM_SMI})
#install(FILES ${PROJECT_SOURCE_DIR}/rocm_smi/python_smi_tools/rocm_smi.py
#       PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
#                         GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
#                                        DESTINATION libexec/${ROCM_SMI})
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bin/rocm-smi
#                                        DESTINATION bin)

# NOTE: Doxygen generation has been stripped
