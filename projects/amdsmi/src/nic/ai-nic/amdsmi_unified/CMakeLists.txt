# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.16)

# Capture the amd_smi project root before the project() call below opens a new
# project scope, which would repoint PROJECT_SOURCE_DIR at this subdirectory.
# Anchoring header paths here avoids counting "../" levels up to the root.
set(AMD_SMI_ROOT_DIR ${PROJECT_SOURCE_DIR})

project(amdsminic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The top-level configure resolved libnl (bundled config targets or pkg-config). Reuse that result:
# a second lookup here can pick a different libnl and load both into libamd_smi.so.
if(USING_BUNDLED_NETLINK OR (LIBNL3_FOUND AND LIBNLGENL3_FOUND))
    set(NIC_HAS_LIBNL TRUE)
endif()

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(NIC_INTERFACE_HEADERS_DIR ${AMD_SMI_ROOT_DIR}/include/amd_smi/impl/nic/amdsmi_unified/interface)
set(NIC_INCLUDE_DIR ${PROJECT_ROOT}/inc ${NIC_INTERFACE_HEADERS_DIR})
set(NIC_SOURCE_DIR ${PROJECT_ROOT}/src)
set(NIC_BUILD_DIR ${CMAKE_BINARY_DIR}/build)
set(NIC_INTERFACE_DIR ${CMAKE_BINARY_DIR}/interface)

set(NIC_DEFAULT_CXX_FLAGS
    "-Wall -Wextra -Werror -Wno-missing-field-initializers -Wno-array-bounds -Wmissing-declarations -Werror=conversion -Wshift-negative-value -fPIC"
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NIC_DEFAULT_CXX_FLAGS}")

# Linker hardening flags - applied only at link time, not during compilation.
# These were previously in CMAKE_CXX_FLAGS which caused -Wunused-command-line-argument
# errors with clang-based compilers (e.g. amdclang).
set(NIC_LINKER_FLAGS "-Wl,-z,relro,-z,noexecstack,-z,noexecheap -Wl,--strip-debug -Wl,--strip-all")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${NIC_LINKER_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${NIC_LINKER_FLAGS}")

# Sources are listed explicitly, like the top-level SRC_LIST, not globbed.
set(CPP_SRCS
    "${NIC_SOURCE_DIR}/smi_ethtool_ioctl.cpp"
    "${NIC_SOURCE_DIR}/smi_nic.cpp"
    "${NIC_SOURCE_DIR}/smi_ibverbs.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_hwmon.cpp"
    "${NIC_SOURCE_DIR}/smi_pci_topology.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_interface.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_log.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_subsystem.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_system.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_telemetry.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_transport_ioctl.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_transport_log.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_vpd.cpp"
    "${NIC_SOURCE_DIR}/smi_sysfs.cpp"
    "${NIC_SOURCE_DIR}/vendors/vendor_registry.cpp"
    "${NIC_SOURCE_DIR}/vendors/amd/ualoe_subsystem.cpp"
    "${NIC_SOURCE_DIR}/vendors/broadcom/broadcom_subsystem.cpp"
    "${NIC_SOURCE_DIR}/vendors/broadcom/broadcom_stats.cpp"
    "${NIC_SOURCE_DIR}/vendors/pensando/pensando_subsystem.cpp"
    "${NIC_SOURCE_DIR}/vendors/pensando/pensando_stats.cpp"
)

# Sources of the netlink transport backend, which requires libnl-3.
set(CPP_SRCS_NETLINK
    "${NIC_SOURCE_DIR}/netlink_generic.cpp"
    "${NIC_SOURCE_DIR}/netlink_wrapper.cpp"
    "${NIC_SOURCE_DIR}/smi_devlink_netlink.cpp"
    "${NIC_SOURCE_DIR}/smi_nic_transport_netlink.cpp"
)

if(NIC_HAS_LIBNL)
    list(APPEND CPP_SRCS ${CPP_SRCS_NETLINK})
    message(STATUS "libnl-3 found: building with netlink transport backend")

    # Kernel UAPI headers older than a feature lack its symbols (Ubuntu 20.04 ships 5.4). Without
    # ethtool netlink, FEC statistics are unavailable and the transport uses ioctl; without the two
    # devlink attributes, the port split state and board serial read N/A.
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles(
        "
        #include <linux/ethtool_netlink.h>
        int main() { return static_cast<int>(ETHTOOL_A_FEC_STAT_CORR_BITS) + ETHTOOL_FLAG_STATS; }"
        NIC_HAS_ETHTOOL_NETLINK
    )
    check_cxx_source_compiles(
        "
        #include <linux/devlink.h>
        int main() { return static_cast<int>(DEVLINK_ATTR_PORT_SPLITTABLE); }"
        NIC_HAS_DEVLINK_PORT_SPLITTABLE
    )
    check_cxx_source_compiles(
        "
        #include <linux/devlink.h>
        int main() { return static_cast<int>(DEVLINK_ATTR_INFO_BOARD_SERIAL_NUMBER); }"
        NIC_HAS_DEVLINK_BOARD_SERIAL
    )

    if(NIC_HAS_ETHTOOL_NETLINK)
        list(APPEND CPP_SRCS "${NIC_SOURCE_DIR}/netlink_ethtool.cpp")
    else()
        message(WARNING "ethtool netlink UAPI header not usable: FEC statistics will be unavailable")
        list(APPEND CPP_SRCS "${NIC_SOURCE_DIR}/netlink_ethtool_stub.cpp")
    endif()
else()
    message(STATUS "libnl-3 not found: building without netlink transport backend")
endif()

add_library(amdsminic STATIC ${CPP_SRCS})
set_target_properties(amdsminic PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${NIC_BUILD_DIR})

# Vendor plugins live under src/vendors/<name>/; expose the vendors root so the
# registry and plugin translation units resolve their headers.
target_include_directories(amdsminic PRIVATE "${NIC_SOURCE_DIR}/vendors")

# libibverbs fills the RDMA port MTU. It is loaded with dlopen at run time, so the headers are only
# needed to build; without them the MTU falls back to the sysfs files, which most drivers lack.
find_path(LIBIBVERBS_INCLUDE_DIR infiniband/verbs.h)
if(LIBIBVERBS_INCLUDE_DIR)
    target_include_directories(amdsminic PRIVATE ${LIBIBVERBS_INCLUDE_DIR})
    target_compile_definitions(amdsminic PRIVATE LIBIBVERBS_INSTALLED)
else()
    message(WARNING "libibverbs headers not found: the RDMA port MTU will read N/A on most drivers")
endif()
target_link_libraries(amdsminic PRIVATE ${CMAKE_DL_LIBS})

# Add compile definition if libnl-3 is available
if(NIC_HAS_LIBNL)
    target_compile_definitions(amdsminic PRIVATE HAVE_LIBNL3)
    # Negative macros: a build that never runs the probes above keeps the full code path.
    if(NOT NIC_HAS_DEVLINK_PORT_SPLITTABLE)
        target_compile_definitions(amdsminic PRIVATE NIC_NO_DEVLINK_PORT_SPLITTABLE)
    endif()
    if(NOT NIC_HAS_DEVLINK_BOARD_SERIAL)
        target_compile_definitions(amdsminic PRIVATE NIC_NO_DEVLINK_BOARD_SERIAL)
    endif()
    if(USING_BUNDLED_NETLINK)
        target_link_libraries(amdsminic PRIVATE libnl::genl)
    else()
        # Absolute paths: bare -lnl-3 names can resolve to a different libnl than the rest of the
        # link uses and load two copies, which crashes at startup (see src/ualoe_lib).
        target_include_directories(amdsminic PRIVATE ${LIBNL3_INCLUDE_DIRS} ${LIBNLGENL3_INCLUDE_DIRS})
        target_link_libraries(amdsminic PRIVATE ${LIBNL3_LINK_LIBRARIES} ${LIBNLGENL3_LINK_LIBRARIES})
    endif()
endif()

target_include_directories(
    amdsminic
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_ROOT}/inc>"
        "$<BUILD_INTERFACE:${NIC_INTERFACE_HEADERS_DIR}>"
        "$<BUILD_INTERFACE:${NIC_INTERFACE_DIR}>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/inc>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/amd_smi/impl/nic/amdsmi_unified/interface>"
)
