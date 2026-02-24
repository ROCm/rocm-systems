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

set(RAS_DECODE "ras-decode")
set(RAS_DECODE_REPO "git@github.amd.com:dctools/ras-decode-instinct-staging.git")
set(RAS_DECODE_SRC_DIR "${PROJECT_SOURCE_DIR}/src/${RAS_DECODE}")
set(RAS_DECODE_INC_DIR "${PROJECT_SOURCE_DIR}/include/${RAS_DECODE}")
find_package(Git QUIET)
if(GIT_FOUND)
    add_custom_TARGET(
        ${RAS_DECODE}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}
        COMMAND ${GIT_EXECUTABLE} clone ${RAS_DECODE_REPO} -b main ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${RAS_DECODE_SRC_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}/src ${RAS_DECODE_SRC_DIR}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${RAS_DECODE_INC_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}/include ${RAS_DECODE_INC_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Cloning ${RAS_DECODE_REPO} from GitHub"
        VERBATIM
    )
else()
    message(WARNING "Git not found, cannot clone ${RAS_DECODE_REPO}.")
endif()
file(GLOB RAS_DECODE_INCLUDES "${RAS_DECODE_INC_DIR}/*.h")
file(GLOB RAS_DECODE_SOURCES "${RAS_DECODE_SRC_DIR}/*.c")
list(REMOVE_ITEM RAS_DECODE_SOURCES "main.c")

set(SRC_LIST ${SRC_LIST} ${RAS_DECODE_SOURCES})
set(INC_LIST ${INC_LIST} ${RAS_DECODE_INCLUDES})

if(ENABLE_ESMI_LIB)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi.h)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi_monitor.h)
    list(APPEND INC_LIST ${ESMI_INC_DIR}/e_smi/e_smi_utils.h)
endif()
# VERSION_* variables should be set by get_version_from_tag
message("Package version: ${PKG_VERSION_STR}")

set(SO_VERSION_STRING "${MAJOR}.${MINOR}.${RELEASE}")
message("SOVERSION: ${SO_VERSION_STRING}")

#
#   --- When BUILD_BOTH_LIBS: build shared and static; tests/examples link to static ---
#   Otherwise: build one library (BUILD_SHARED_LIBS).
#
if(BUILD_BOTH_LIBS)
    add_library(${AMD_SMI} SHARED ${SRC_LIST} ${INC_LIST})
    add_library(${AMD_SMI}_static STATIC ${SRC_LIST} ${INC_LIST})
    set(AMD_SMI_TARGET ${AMD_SMI}_static PARENT_SCOPE)
    set(AMD_SMI_EXPORT_TARGETS ${AMD_SMI} ${AMD_SMI}_static PARENT_SCOPE)
    set(_AMD_SMI_ALL_TARGETS ${AMD_SMI} ${AMD_SMI}_static)
else()
    add_library(${AMD_SMI} ${SRC_LIST} ${INC_LIST})
    set(AMD_SMI_TARGET ${AMD_SMI} PARENT_SCOPE)
    set(AMD_SMI_EXPORT_TARGETS ${AMD_SMI} PARENT_SCOPE)
    set(_AMD_SMI_ALL_TARGETS ${AMD_SMI})
endif()

foreach(_TARGET IN LISTS _AMD_SMI_ALL_TARGETS)
    target_link_libraries(${_TARGET} PRIVATE
            rt
            Threads::Threads
            ${CMAKE_DL_LIBS}
            ${FILESYSTEM_LIB}
        )
        target_include_directories(${_TARGET} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/rocm_smi/include
            ${PROJECT_SOURCE_DIR}/common/shared_mutex
            ${RAS_DECODE_INC_DIR}
            ${DRM_INCLUDE_DIRS}
            ${DRM_AMDGPU_INCLUDE_DIRS}
        )
        target_include_directories(${_TARGET} PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
        )
endforeach()

#
#   --- When building the shared library ---
#   hide all symbols by default; only AMDSMI_EXPORT in amdsmi.h are exported
#
if(BUILD_BOTH_LIBS OR BUILD_SHARED_LIBS)
    target_compile_options(${AMD_SMI} PRIVATE -fvisibility=hidden)
    target_compile_definitions(${AMD_SMI} PRIVATE AMDSMI_BUILD_LIBRARY)
endif()

## Set the VERSION and SOVERSION values (shared library only)
set_property(TARGET ${AMD_SMI} PROPERTY SOVERSION "${MAJOR}")
set_property(TARGET ${AMD_SMI} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the shared target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(BUILD_BOTH_LIBS OR ${BUILD_SHARED_LIBS})
        if(TARGET ${AMD_SMI})
            add_custom_command(TARGET ${AMD_SMI} POST_BUILD COMMAND ${CMAKE_STRIP} lib${AMD_SMI}.so.${SO_VERSION_STRING})
        endif()
    endif()
endif()

## Add the install directives for the runtime library.
set(_install_TARGETs ${AMD_SMI})
if(BUILD_BOTH_LIBS)
    list(APPEND _install_TARGETs ${AMD_SMI}_static)
endif()
install(
    TARGETS ${_install_TARGETs}
    EXPORT amd_smiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev)
install(
    TARGETS ${AMD_SMI}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

install(
    FILES ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi
    COMPONENT dev)
