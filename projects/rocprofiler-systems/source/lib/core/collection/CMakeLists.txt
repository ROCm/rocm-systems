# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(collection_headers
    ${CMAKE_CURRENT_LIST_DIR}/sample_batch.hpp
    ${CMAKE_CURRENT_LIST_DIR}/schema.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sink_interface.hpp
)

target_sources(rocprofiler-systems-core-library PRIVATE ${collection_headers})
