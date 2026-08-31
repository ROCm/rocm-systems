# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# UALOE Library CMake configuration

# Netlink dependencies (LIBNL3/LIBNLGENL3/LIBMNL) are resolved once by the
# top-level CMakeLists.txt via pkg_check_modules; its result variables
# (*_CFLAGS, *_LINK_LIBRARIES) are reused below. Do not re-run pkg_check_modules
# here: raw *_LIBRARIES yield bare -lnl-3/-lmnl with no -L, which fails to link
# when the libs live outside the default linker path (e.g. rocm_sysdeps).

# Define source files for ualoe library
set(UALOE_SRC_LIST
    ualoe_lib.c
    ualoe_cdev.c
    ualoe_cb.c
    ualoe_nl.c
    ualoe_telem.c
)

# Convert relative paths to absolute paths
set(UALOE_ABS_SRC_LIST)
foreach(src ${UALOE_SRC_LIST})
    list(APPEND UALOE_ABS_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
endforeach()

# Define header files
set(UALOE_HDR_LIST
    "${UALOE_LIB_INC_DIR}/ualoe_lib.h"
    "${UALOE_LIB_INC_DIR}/ualoe_cdev.h"
    "${UALOE_LIB_INC_DIR}/ualoe_cb.h"
    "${UALOE_LIB_INC_DIR}/ualoe_nl.h"
    "${UALOE_LIB_INC_DIR}/ifoe_telemetry.h"
    "${UALOE_LIB_INC_DIR}/ifoe_telemetry_hostapp.h"
    "${UALOE_LIB_INC_DIR}/cbl_cfg/uapi.h"
)

# Export variables to parent scope so they can be used by src/CMakeLists.txt
set(UALOE_SOURCES ${UALOE_ABS_SRC_LIST} PARENT_SCOPE)
set(UALOE_HEADERS ${UALOE_HDR_LIST} PARENT_SCOPE)
set(UALOE_INCLUDE_DIR ${UALOE_LIB_INC_DIR} PARENT_SCOPE)

# Export dependency information to parent scope - make dependencies hard requirements
if(USING_BUNDLED_NETLINK)
    # Using bundled sysdeps with CMake targets
    set(UALOE_COMPILE_FLAGS "-DUALOE_NETLINK" PARENT_SCOPE)
    set(UALOE_LINK_LIBRARIES "libnl::genl;libmnl::libmnl" PARENT_SCOPE)
    set(UALOE_DEPENDENCIES_FOUND TRUE PARENT_SCOPE)
    message(STATUS "UALOE using bundled netlink libraries")
elseif(LIBNL3_FOUND AND LIBNLGENL3_FOUND AND LIBMNL_FOUND)
    # Link via pkg-config absolute library paths (*_LINK_LIBRARIES) rather than
    # bare -lnl-3/-lmnl: the absolute paths resolve without a -L search dir when
    # the libs live outside the default linker path (e.g. rocm_sysdeps), and
    # unlike PkgConfig:: imported targets they remain valid in the exported
    # amd_smi_static link interface for downstream find_package() consumers.
    set(UALOE_COMPILE_FLAGS "${LIBNL3_CFLAGS} ${LIBNLGENL3_CFLAGS} ${LIBMNL_CFLAGS} -DUALOE_NETLINK" PARENT_SCOPE)
    set(UALOE_LINK_LIBRARIES
        "${LIBNL3_LINK_LIBRARIES};${LIBNLGENL3_LINK_LIBRARIES};${LIBMNL_LINK_LIBRARIES}"
        PARENT_SCOPE
    )
    set(UALOE_DEPENDENCIES_FOUND TRUE PARENT_SCOPE)
    message(STATUS "UALOE using system netlink libraries")
else()
    message(
        FATAL_ERROR
        "Required UALOE libraries not found: libnl-3.0, libnl-genl-3.0, or libmnl. These are mandatory dependencies."
    )
    set(UALOE_DEPENDENCIES_FOUND FALSE PARENT_SCOPE)
endif()
