# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

add_executable(rj_ip_discovery main.cpp)
target_link_libraries(rj_ip_discovery PRIVATE rocjitsu_pci simdojo util)
set_target_properties(rj_ip_discovery PROPERTIES OUTPUT_NAME rj-ip-discovery)
rj_configure_target(rj_ip_discovery TOOL)
