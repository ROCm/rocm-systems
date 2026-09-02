## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT
set(target compression)

add_library(${target} STATIC gzip_output_stream.cpp gzip_output_stream.h)

# Consumers include "compression/<header>.h", so the parent directory is what
# goes on the include path.
target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# PRIVATE: hide zlib headers; static lib still needs $<LINK_ONLY:ZLIB::ZLIB>.
target_link_libraries(${target} PRIVATE ZLIB::ZLIB)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()
