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

option(BRCM_NIC "Build brcm-nic" OFF)
if(BRCM_NIC)
    add_compile_definitions(BRCM_NIC)
    set(NIC_SOURCES
        "${PROJECT_SOURCE_DIR}/src/nic/brcm-nic/amd_smi_lspci_commands.cc"
        "${PROJECT_SOURCE_DIR}/src/nic/brcm-nic/amd_smi_nic_device.cc"
        "${PROJECT_SOURCE_DIR}/src/nic/brcm-nic/amd_smi_no_drm_nic.cc"
        "${PROJECT_SOURCE_DIR}/src/nic/brcm-nic/amd_smi_no_drm_switch.cc"
        "${PROJECT_SOURCE_DIR}/src/nic/brcm-nic/amd_smi_switch_device.cc"
    )
    set(NIC_INCLUDES
        ${INC_DIR}/impl/nic/amd_smi_lspci_commands.h
        ${INC_DIR}/impl/nic/amd_smi_nic_device.h
        ${INC_DIR}/impl/nic/amd_smi_no_drm_nic.h
        ${INC_DIR}/impl/nic/amd_smi_no_drm_switch.h
        ${INC_DIR}/impl/nic/amd_smi_switch_device.h
    )
endif()

set(NIC_SOURCES ${NIC_SOURCES} "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amd_smi_ainic_device.cc")
set(NIC_INCLUDES ${NIC_INCLUDES} ${INC_DIR}/impl/nic/amd_smi_ainic_device.h)
set(NIC_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/")

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
    "${CMN_SRC_LIST}"
    "${NIC_SOURCES}"
)

file(GLOB IMPL_HEADERS "${INC_DIR}/impl/*.h")
set(INC_LIST
    "${IMPL_HEADERS}"
    "${INC_DIR}/amdsmi.h"
    "${PROJECT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/rocm_smi.h"
    "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/rocm_smi_utils.h"
    "${NIC_INCLUDES}"
)

set(RAS_DECODE "ras-decode")
set(RAS_DECODE_REPO "git@github.amd.com:dctools/ras-decode-instinct-staging.git")
set(RAS_DECODE_SRC_DIR "${PROJECT_SOURCE_DIR}/src/${RAS_DECODE}")
set(RAS_DECODE_INC_DIR "${PROJECT_SOURCE_DIR}/include/${RAS_DECODE}")
find_package(Git QUIET)
if(GIT_FOUND)
    add_custom_target(
        ${RAS_DECODE}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}
        COMMAND ${GIT_EXECUTABLE} clone ${RAS_DECODE_REPO} -b main ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${RAS_DECODE_SRC_DIR}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}/src ${RAS_DECODE_SRC_DIR}
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${RAS_DECODE_INC_DIR}
        COMMAND
            ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/${RAS_DECODE}/include ${RAS_DECODE_INC_DIR}
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

add_subdirectory(nic/ai-nic/amdsmi_unified)

#
#   --- When BUILD_BOTH_LIBS: build shared and static; tests/examples link to static ---
#   Otherwise: build one library (BUILD_SHARED_LIBS).
#
if(BUILD_BOTH_LIBS)
    add_library(${AMD_SMI} SHARED ${SRC_LIST} ${INC_LIST})
    add_library(${AMD_SMI}_static STATIC ${SRC_LIST} ${INC_LIST})
    set(AMD_SMI_TARGET ${AMD_SMI}_static PARENT_SCOPE)
    set(AMD_SMI_EXPORT_TARGETS ${AMD_SMI} ${AMD_SMI}_static amdsminic PARENT_SCOPE)
    set(_AMD_SMI_ALL_TARGETS ${AMD_SMI} ${AMD_SMI}_static)
else()
    add_library(${AMD_SMI} ${SRC_LIST} ${INC_LIST})
    set(AMD_SMI_TARGET ${AMD_SMI} PARENT_SCOPE)
    set(AMD_SMI_EXPORT_TARGETS ${AMD_SMI} amdsminic PARENT_SCOPE)
    set(_AMD_SMI_ALL_TARGETS ${AMD_SMI})
endif()

foreach(_TARGET IN LISTS _AMD_SMI_ALL_TARGETS)
    target_link_libraries(
        ${_TARGET}
        PRIVATE rt Threads::Threads ${CMAKE_DL_LIBS} amdsminic ${FILESYSTEM_LIB}
    )
    target_include_directories(
        ${_TARGET}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/../
            ${PROJECT_SOURCE_DIR}/rocm_smi/include
            ${PROJECT_SOURCE_DIR}/common/shared_mutex
            ${RAS_DECODE_INC_DIR}
            ${DRM_INCLUDE_DIRS}
            ${DRM_AMDGPU_INCLUDE_DIRS}
            ${NIC_INCLUDE_DIRS}
    )
    target_include_directories(
        ${_TARGET}
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nic/ai-nic/amdsmi_unified/interface>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )
    target_link_directories(${_TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/nic/ai-nic/amdsmi_unified/build/)
endforeach()

#
#   --- When building the shared library ---
#   Use a linker version script to export only the public C API (amdsmi_*).
#   All other symbols (including internal C++ amd::smi::* and rsmi_*) are hidden
#   by the version script's "local: *;" directive at link time.
#   This prevents symbol collisions when libamd_smi.so and librocm_smi64.so
#   are loaded in the same process (see TheRock#3487).
#
if(BUILD_BOTH_LIBS OR BUILD_SHARED_LIBS)
    set(VERSION_SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/libamd_smi.map")
    file(
        WRITE ${VERSION_SCRIPT}
        "# All public API functions MUST use the amdsmi_ prefix\n"
        "# or be explicitly listed below to be exported.\n"
        "AMDSMI_1 {\n"
        "    global:\n"
        "        amdsmi_*;\n"
        "        # rsmi_init and rsmi_is_P2P_accessible are kept visible\n"
        "        # for external libs that already depend on them\n"
        "        # through libamd_smi.so (e.g. rocm_smi compatibility layer).\n"
        "        rsmi_init;\n"
        "        rsmi_is_P2P_accessible;\n"
        "    local:\n"
        "        *;\n"
        "};\n"
    )
    target_link_options(${AMD_SMI} PRIVATE "LINKER:--version-script=${VERSION_SCRIPT}")
    set_property(TARGET ${AMD_SMI} APPEND PROPERTY LINK_DEPENDS "${VERSION_SCRIPT}")
endif()

## Set the VERSION and SOVERSION values
set_property(TARGET ${AMD_SMI} PROPERTY SOVERSION "${MAJOR}")
set_property(TARGET ${AMD_SMI} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    if(BUILD_BOTH_LIBS OR BUILD_SHARED_LIBS)
        if(TARGET ${AMD_SMI})
            add_custom_command(
                TARGET ${AMD_SMI}
                POST_BUILD
                COMMAND ${CMAKE_STRIP} lib${AMD_SMI}.so.${SO_VERSION_STRING}
            )
        endif()
    endif()
endif()

## Add the install directives for the runtime library.
set(_INSTALL_TARGETS ${AMD_SMI} amdsminic)
if(BUILD_BOTH_LIBS)
    list(APPEND _INSTALL_TARGETS ${AMD_SMI}_static)
endif()

install(
    TARGETS ${_INSTALL_TARGETS}
    EXPORT amd_smiTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev
)

install(TARGETS ${AMD_SMI} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)

install(
    FILES ${PROJECT_SOURCE_DIR}/include/amd_smi/amdsmi.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi
    COMPONENT dev
)

install(FILES ${IMPL_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/ COMPONENT dev)

install(FILES ${NIC_INCLUDES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/ COMPONENT dev)

set(UNIFIED_NIC_INCLUDES ${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/interface/smi_nic_interface.h)
install(FILES ${UNIFIED_NIC_INCLUDES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/ COMPONENT dev)
install(
    FILES ${UNIFIED_NIC_INCLUDES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/interface
    COMPONENT dev
)
set(UNIFIED_NIC_INCLUDES
    "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/inc/smi_devlink_netlink.h"
    "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/inc/smi_ethtool_ioctl.h"
    "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/inc/smi_nic.h"
    "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/inc/smi_nic_subsystem.h"
    "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/inc/smi_nic_system.h"
    "${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/inc/smi_sysfs.h"
)
install(
    FILES ${UNIFIED_NIC_INCLUDES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/inc/
    COMPONENT dev
)

file(GLOB ESMI_HEADERS "${PROJECT_SOURCE_DIR}/esmi_ib_library/include/e_smi/*.h")
install(FILES ${ESMI_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/e_smi/ COMPONENT dev)

install(
    FILES "${PROJECT_SOURCE_DIR}/rocm_smi/include/rocm_smi/rocm_smi_logger.h"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocm_smi/
    COMPONENT dev
)
