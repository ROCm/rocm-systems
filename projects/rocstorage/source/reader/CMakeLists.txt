# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ------------------------------------------------------------ #
# Reader Data Model Sources (from rocstorage-reader)
# ------------------------------------------------------------ #

set(READER_DATAMODEL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_c_interface.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_base.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_track.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_track_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_event_track_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_pmc_track_slice.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_flow_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_stack_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_ext_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel/rocprofvis_dm_table_row.cpp
)

# ------------------------------------------------------------ #
# Reader Database Sources (from rocstorage-reader)
# ------------------------------------------------------------ #

set(READER_DATABASE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db_sqlite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db_rocpd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db_rocprof.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db_profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db_future.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db_query_factory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/rocprofvis_db_query_builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/database/unified_db_adapter.cpp
)

# ------------------------------------------------------------ #
# Add sources to rocstorage library
# ------------------------------------------------------------ #

target_sources(rocstorage PRIVATE
    ${READER_DATAMODEL_SOURCES}
    ${READER_DATABASE_SOURCES}
)

# ------------------------------------------------------------ #
# Include directories for reader sources
# ------------------------------------------------------------ #

target_include_directories(rocstorage PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/datamodel
    ${CMAKE_CURRENT_SOURCE_DIR}/database
    ${CMAKE_CURRENT_SOURCE_DIR}/interop
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/source
)

# ------------------------------------------------------------ #
# Compile definitions
# ------------------------------------------------------------ #

target_compile_definitions(rocstorage PRIVATE CPPBUILD)
