message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  CMake AMD SMI C/C++ Library                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})

## Include common cmake modules
include(utils)

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
    "${SRC_DIR}/scoped_fd.cc"
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
    "${INC_DIR}/impl/amd_smi_utils.h"
    "${INC_DIR}/impl/amd_smi_uuid.h"
    "${INC_DIR}/impl/scoped_fd.h"
    "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/rocm_smi.h"
    "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/rocm_smi_utils.h")

set(ACA_SRC_DIR "aca-decode")
set(SRC_LIST ${SRC_LIST} ${ACA_SRC_DIR}/aca_api.c ${ACA_SRC_DIR}/aca_decode.c ${ACA_SRC_DIR}/aca_fields.c
             ${ACA_SRC_DIR}/aca_tables.c ${ACA_SRC_DIR}/error_map.c)
set(ACA_INC_DIR "${PROJECT_SOURCE_DIR}/include/aca-decode")
set(INC_LIST ${INC_LIST} ${ACA_INC_DIR}/aca_decode.h ${ACA_INC_DIR}/aca_fields.h ${ACA_INC_DIR}/aca_tables.h
             ${ACA_INC_DIR}/error_map.h)

if(ENABLE_ESMI_LIB)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi.h)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi_monitor.h)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi_utils.h)
endif()
# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

set(SO_VERSION_STRING "${MAJOR}.${MINOR}.${RELEASE}")
message("SOVERSION: ${SO_VERSION_STRING}")

add_library(${AMD_SMI} ${SRC_LIST} ${INC_LIST})
target_link_libraries(${AMD_SMI} PRIVATE
    rt
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
target_include_directories(${AMD_SMI} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/rocm_smi/include
    ${PROJECT_SOURCE_DIR}/common/shared_mutex 
    ${ACA_INC_DIR}
    ${DRM_INCLUDE_DIRS}
    ${DRM_AMDGPU_INCLUDE_DIRS}                                          
)

# use the target_include_directories() command to specify the include directories for the target
target_include_directories(${AMD_SMI} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                                             "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

## Set the VERSION and SOVERSION values
set_property(TARGET ${AMD_SMI} PROPERTY SOVERSION "${MAJOR}")
set_property(TARGET ${AMD_SMI} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(${BUILD_SHARED_LIBS}) #stripping only for .so
        add_custom_command(TARGET ${AMD_SMI} POST_BUILD COMMAND ${CMAKE_STRIP} lib${AMD_SMI}.so.${SO_VERSION_STRING})
    endif()
endif()

## Add the install directives for the runtime library.
install(
    TARGETS ${AMD_SMI}
    EXPORT amd_smiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev)

install(
    TARGETS ${AMD_SMI}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

install(
    FILES ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi
    COMPONENT dev)
