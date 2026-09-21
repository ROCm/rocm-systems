# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_subdirectory(kfd)

set(buffered_domains_headers ${kfd_domains_headers})

set(buffered_domains_headers ${buffered_domains_headers} PARENT_SCOPE)
