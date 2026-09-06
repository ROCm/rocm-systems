# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  CMake AMD SMI C/C++ Library                      ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})

## Include common cmake modules
include(utils)

# BUILD_PYTHON_WHEEL is declared at the project root (CMakeLists.txt)
# so a single -D switch propagates to every subdirectory regardless of
# the order add_subdirectory() calls run in. Do NOT redeclare it here.

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
    # UMA carveout adapter: talks to the fwupd daemon over D-Bus by loading
    # libdbus-1 at runtime, so it adds no build- or link-time dependency.
    "${SRC_DIR}/fwupd_carveout.cc"
    "${SRC_DIR}/amd_smi_cper.cc"
    "${SRC_DIR}/amd_smi_common.cc"
    "${SRC_DIR}/amd_smi_drm.cc"
    "${SRC_DIR}/amd_smi_fabric_ualink.cc"
    "${SRC_DIR}/amd_smi_gpu_device.cc"
    "${SRC_DIR}/amd_smi_lib_loader.cc"
    "${SRC_DIR}/amd_smi_socket.cc"
    "${SRC_DIR}/amd_smi_system.cc"
    "${SRC_DIR}/amd_smi_ualoe.cc"
    "${SRC_DIR}/amd_smi_utils.cc"
    "${SRC_DIR}/amd_smi_uuid.cc"
    "${SRC_DIR}/scoped_fd.cc"
    "${SRC_DIR}/fdinfo.cc"
    "${CMN_SRC_LIST}"
    "${NIC_SOURCES}"
)

if(ENABLE_WSL_BACKEND)
    list(APPEND SRC_LIST "${SRC_DIR}/amd_smi_wsl_device.cc")
endif()

# CONFIGURE_DEPENDS re-evaluates the glob at build time so adding or removing an
# impl header triggers a regenerate instead of going unnoticed until a clean build.
file(GLOB IMPL_HEADERS CONFIGURE_DEPENDS "${INC_DIR}/impl/*.h")
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

# Include UALOE library - always required
set(UALOE_LIB "ualoe_lib")
set(UALOE_LIB_SRC_DIR "${PROJECT_SOURCE_DIR}/src/${UALOE_LIB}")
set(UALOE_LIB_INC_DIR "${PROJECT_SOURCE_DIR}/include/${UALOE_LIB}")
add_subdirectory("ualoe_lib")
set(SRC_LIST ${SRC_LIST} ${UALOE_SOURCES})
set(INC_LIST ${INC_LIST} ${UALOE_HEADERS})

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

option(BUILD_CUID "Build CUID library" OFF)

if(BUILD_CUID)
    set(ROCM_DIR_CORE "/opt/rocm/core")
    # include CUID library
    find_library(
        AMDCUID_LIB
        NAMES libamdcuid_static.a amdcuid_static
        HINTS ${ROCM_DIR_CORE}/lib ${CMAKE_PREFIX_PATH}
        REQUIRED
    )
    find_path(
        AMDCUID_INCLUDE_DIR
        NAMES amd_cuid.h
        HINTS ${ROCM_DIR_CORE}/include/amdcuid ${CMAKE_PREFIX_PATH}/include/amdcuid
        REQUIRED
    )
endif(BUILD_CUID)
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

# Apply the include dirs, link libraries and link dirs shared by every amd_smi
# library target (the main/static libraries and the wheel-only
# libamd_smi_python). Target-specific settings -- version script, SOVERSION,
# -Bsymbolic-functions, the extra NIC interface includes -- are applied by the
# caller after this.
function(amdsmi_configure_common_target _TARGET)
    target_link_libraries(
        ${_TARGET}
        PRIVATE rt Threads::Threads ${CMAKE_DL_LIBS} amdsminic ${FILESYSTEM_LIB} ${UALOE_LINK_LIBRARIES}
    )

    if(ENABLE_WSL_BACKEND)
        target_compile_definitions(${_TARGET} PRIVATE ENABLE_WSL_BACKEND)
        set(_WSL_HSAKMT_INC "${PROJECT_SOURCE_DIR}/../rocr-runtime/libhsakmt/include")
        if(NOT EXISTS "${_WSL_HSAKMT_INC}/hsakmt/rocdxg_smi.h")
            message(
                FATAL_ERROR
                "ENABLE_WSL_BACKEND=ON requires ${_WSL_HSAKMT_INC}/hsakmt/rocdxg_smi.h "
                "(built from rocr-runtime/libhsakmt's WIN_SDK/dxg path). "
                "Check out and build the matching libhsakmt tree before enabling this flag."
            )
        endif()
        target_include_directories(${_TARGET} PRIVATE ${_WSL_HSAKMT_INC})
    endif()

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
            ${UALOE_INCLUDE_DIR}
    )
    target_include_directories(
        ${_TARGET}
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )

    if(BUILD_CUID)
        target_compile_definitions(${_TARGET} PRIVATE BUILD_CUID)
        target_link_libraries(${_TARGET} PRIVATE ${AMDCUID_LIB})
        target_include_directories(${_TARGET} PRIVATE ${AMDCUID_INCLUDE_DIR})
    endif()

    target_link_directories(${_TARGET} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/nic/ai-nic/amdsmi_unified/build/)
endfunction()

foreach(_TARGET IN LISTS _AMD_SMI_ALL_TARGETS)
    amdsmi_configure_common_target(${_TARGET})
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

# Add UALOE compile flags for netlink dependencies (e.g. -I.../libnl3 for
# <netlink/*.h>). Must apply to every amd_smi target: with BUILD_BOTH_LIBS the
# static library is built separately and would otherwise miss these includes.
if(UALOE_COMPILE_FLAGS)
    separate_arguments(UALOE_FLAGS NATIVE_COMMAND "${UALOE_COMPILE_FLAGS}")
    foreach(_TARGET IN LISTS _AMD_SMI_ALL_TARGETS)
        target_compile_options(${_TARGET} PRIVATE ${UALOE_FLAGS})
    endforeach()
endif()

## Set the VERSION and SOVERSION values
set_property(TARGET ${AMD_SMI} PROPERTY SOVERSION "${MAJOR}")
set_property(TARGET ${AMD_SMI} PROPERTY VERSION "${SO_VERSION_STRING}")

## If the library is a release, strip the target library
if("${CMAKE_BUILD_TYPE}" STREQUAL Release AND CMAKE_STRIP)
    if(BUILD_BOTH_LIBS OR BUILD_SHARED_LIBS)
        if(TARGET ${AMD_SMI})
            add_custom_command(TARGET ${AMD_SMI} POST_BUILD COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${AMD_SMI}>)
        endif()
    endif()
endif()

## Create Python-specific library if BUILD_PYTHON_WHEEL is enabled
if(BUILD_PYTHON_WHEEL)
    add_library(${AMD_SMI}_python ${SRC_LIST} ${INC_LIST})
    amdsmi_configure_common_target(${AMD_SMI}_python)
    # Extra NIC interface includes the wheel target needs on top of the common
    # set applied above.
    target_include_directories(
        ${AMD_SMI}_python
        PRIVATE
            ${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/inc
            ${PROJECT_SOURCE_DIR}/src/nic/ai-nic/amdsmi_unified/interface
    )
    # Apply the same UALOE netlink compile flags (e.g. -I.../libnl3 for
    # <netlink/*.h> and -DUALOE_NETLINK) that the other amd_smi targets receive;
    # this target is created after the shared _AMD_SMI_ALL_TARGETS loop above and
    # would otherwise miss them, breaking the ualoe_lib netlink sources.
    if(UALOE_COMPILE_FLAGS)
        separate_arguments(UALOE_FLAGS_PYTHON NATIVE_COMMAND "${UALOE_COMPILE_FLAGS}")
        target_compile_options(${AMD_SMI}_python PRIVATE ${UALOE_FLAGS_PYTHON})
    endif()
    set_property(TARGET ${AMD_SMI}_python PROPERTY SOVERSION "${MAJOR}")
    set_property(TARGET ${AMD_SMI}_python PROPERTY VERSION "${SO_VERSION_STRING}")
    set_target_properties(${AMD_SMI}_python PROPERTIES OUTPUT_NAME "amd_smi_python")
    # -Bsymbolic-functions binds this library's own amdsmi_* calls internally at
    # link time. Without it, when another loaded module already exports the same
    # amdsmi_*@AMDSMI_1 symbols (e.g. PyTorch's bundled libamd_smi.so), the
    # dynamic linker interposes our internal calls onto that other, separately
    # initialized copy -- so amdsmi_init() runs against our state while
    # enumeration reads the other lib's, returning 0 sockets/processors.
    target_link_options(
        ${AMD_SMI}_python
        PRIVATE "LINKER:--version-script=${VERSION_SCRIPT}" "LINKER:-Bsymbolic-functions"
    )
    set_property(TARGET ${AMD_SMI}_python APPEND PROPERTY LINK_DEPENDS "${VERSION_SCRIPT}")
    ## Strip the wheel-shipped library in Release so the published wheel does
    ## not carry DWARF debug symbols (mirrors the system-library strip above).
    if("${CMAKE_BUILD_TYPE}" STREQUAL Release AND CMAKE_STRIP)
        add_custom_command(TARGET ${AMD_SMI}_python POST_BUILD COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${AMD_SMI}_python>)
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

# Impl headers declaring nothing libamd_smi.so exports are build-only: a
# dev-package consumer that included one would hit a link error. Production and
# the unit tests include them from the source tree instead. The version script
# exports the amdsmi_* glob, which matches neither an unprefixed symbol
# (cper_set_read_fn_for_testing) nor a mangled C++ one (amdsmi_test_sleep, and
# both of amd_smi_cper.h's declarations).
set(INSTALL_IMPL_HEADERS ${IMPL_HEADERS})
list(FILTER INSTALL_IMPL_HEADERS EXCLUDE REGEX "_test")
list(FILTER INSTALL_IMPL_HEADERS EXCLUDE REGEX "amd_smi_container_id_parser")
list(FILTER INSTALL_IMPL_HEADERS EXCLUDE REGEX "amd_smi_cper")
list(FILTER INSTALL_IMPL_HEADERS EXCLUDE REGEX "amd_smi_wsl_device")
list(FILTER INSTALL_IMPL_HEADERS EXCLUDE REGEX "amd_smi_wsl_syms")
install(FILES ${INSTALL_IMPL_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/ COMPONENT dev)

install(FILES ${NIC_INCLUDES} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/ COMPONENT dev)

set(UNIFIED_NIC_INCLUDES ${PROJECT_SOURCE_DIR}/include/amd_smi/impl/nic/amdsmi_unified/interface/smi_nic_interface.h)
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
