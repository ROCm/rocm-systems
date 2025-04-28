#
# Minimum version of cmake required
#

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  CMake AMD SMI C/C++ Library                      ")
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

## Include common cmake modules
include(utils)

################# Determine the library version #########################
set(SO_VERSION_GIT_TAG_PREFIX "amdsmi_so_ver")

set(SRC_DIR "amd_smi")
set(INC_DIR "${PROJECT_SOURCE_DIR}/include/amd_smi")

set(SRC_LIST
    "${SRC_DIR}/amd_smi.cc"
    "${SRC_DIR}/amd_smi_cper.cc"
    "${SRC_DIR}/amd_smi_common.cc"
    "${SRC_DIR}/amd_smi_drm.cc"
    "${SRC_DIR}/amd_smi_gpu_device.cc"
    "${SRC_DIR}/amd_smi_lib_loader.cc"
    "${SRC_DIR}/amd_smi_socket.cc"
    "${SRC_DIR}/amd_smi_system.cc"
    "${SRC_DIR}/amd_smi_utils.cc"
    "${SRC_DIR}/amd_smi_uuid.cc"
    "${SRC_DIR}/fdinfo.cc"
    "${CMN_SRC_LIST}")
set(INC_LIST
    "${INC_DIR}/amdsmi.h"
    "${INC_DIR}/impl/amd_smi_common.h"
    "${INC_DIR}/impl/amd_smi_cper.h"
    "${INC_DIR}/impl/amd_smi_processor.h"
    "${INC_DIR}/impl/amd_smi_drm.h"
    "${INC_DIR}/impl/amd_smi_gpu_device.h"
    "${INC_DIR}/impl/amd_smi_lib_loader.h"
    "${INC_DIR}/impl/amd_smi_socket.h"
    "${INC_DIR}/impl/amd_smi_system.h"
    "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/rocm_smi.h"
    "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/rocm_smi_utils.h")

set(ACA_SRC_DIR "aca-decode")
set(ACA_INC_DIR "${PROJECT_SOURCE_DIR}/include/aca-decode")
set(SRC_LIST
    ${SRC_LIST}
    ${ACA_SRC_DIR}/aca_decode.c
    ${ACA_SRC_DIR}/aca_fields.c
    ${ACA_SRC_DIR}/aca_tables.c
    ${ACA_SRC_DIR}/error_map.c
)
set(INC_LIST
    ${INC_LIST}
    ${ACA_INC_DIR}/aca_decode.h
    ${ACA_INC_DIR}/aca_fields.h
    ${ACA_INC_DIR}/aca_tables.h
    ${ACA_INC_DIR}/error_map.h
)

if(ENABLE_ESMI_LIB)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi.h)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi_monitor.h)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi_utils.h)
endif()
# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

# Debian package specific variables
# Set a default value for the package version
get_version_from_tag("${${AMD_SMI_LIBS_TARGET}_VERSION_MAJOR}.${${AMD_SMI_LIBS_TARGET}_VERSION_MINOR}.0.0" ${SO_VERSION_GIT_TAG_PREFIX} GIT)

# VERSION_* variables should be set by get_version_from_tag
if(${ROCM_PATCH_VERSION})
    set(VERSION_PATCH ${ROCM_PATCH_VERSION})
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")
else()
    set(SO_VERSION_STRING "${VERSION_MAJOR}.${VERSION_MINOR}")
endif()
set(${AMD_SMI}_VERSION_MAJOR "${VERSION_MAJOR}")
set(${AMD_SMI}_VERSION_MINOR "${VERSION_MINOR}")
set(${AMD_SMI}_VERSION_PATCH "0")
set(${AMD_SMI}_VERSION_BUILD "0")
message("SOVERSION: ${SO_VERSION_STRING}")

# Create a configure file to get version info from within library
configure_file("${AMD_SMI_TARGET}Config.in" "${PROJECT_SOURCE_DIR}/include/amd_smi/${AMD_SMI_TARGET}Config.h")

add_executable(amd_smi_ex "../example/amd_smi_drm_example.cc")
target_link_libraries(amd_smi_ex ${AMD_SMI_TARGET})
add_library(${AMD_SMI_TARGET} ${SRC_LIST} ${INC_LIST})
target_link_libraries(${AMD_SMI_TARGET} pthread rt dl ${DRM_LIBRARIES} ${AMDGPU_DRM_LIBRARIES})
target_include_directories(${AMD_SMI_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/rocm_smi/include
                                                     ${ACA_INC_DIR}
                                                     ${PROJECT_SOURCE_DIR}/common/shared_mutex)

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${AMD_SMI_TARGET} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                                    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

## Set the VERSION and SOVERSION values
set_property(TARGET ${AMD_SMI_TARGET} PROPERTY SOVERSION "${VERSION_MAJOR}")
set_property(TARGET ${AMD_SMI_TARGET} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(${BUILD_SHARED_LIBS}) #stripping only for .so
        add_custom_command(
            TARGET ${AMD_SMI_TARGET}
            POST_BUILD
            COMMAND ${CMAKE_STRIP} lib${AMD_SMI_TARGET}.so.${SO_VERSION_STRING})
    endif()
endif()

#file reorganization changes
#amd_smi.py moved to libexec/amd_smi. so creating amd-smi symlink
#file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
#add_custom_target(
#    link-amd-smi ALL
#    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#    COMMAND ${CMAKE_COMMAND} -E create_symlink ../libexec/${AMD_SMI}/amd_smi.py ${CMAKE_CURRENT_BINARY_DIR}/bin/amd-smi)

## Add the install directives for the runtime library.
install(
    TARGETS ${AMD_SMI_TARGET}
    EXPORT amd_smiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev)
    #COMPONENT ${AMD_SMI_COMPONENT})

install(
    TARGETS ${AMD_SMI_TARGET}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT asan)

install(
    FILES ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi
    COMPONENT dev)
install(
    FILES ${PROJECT_SOURCE_DIR}/include/amd_smi/${AMD_SMI_TARGET}Config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi
    COMPONENT dev)
install(
    FILES ${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/kfd_ioctl.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi
    COMPONENT dev)
#install(FILES ${PROJECT_SOURCE_DIR}/rocm_smi/python_smi_tools/rsmiBindings.py
#                                        DESTINATION libexec/${AMD_SMI})
#install(FILES ${PROJECT_SOURCE_DIR}/rocm_smi/python_smi_tools/rocm_smi.py
#       PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
#                         GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
#                                        DESTINATION libexec/${AMD_SMI})
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bin/rocm-smi
#                                        DESTINATION bin)
