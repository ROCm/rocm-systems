################################################################################
##
## Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
##
## MIT LICENSE:
## Permission is hereby granted, free of charge, to any person obtaining a copy of
## this software and associated documentation files (the "Software"), to deal in
## the Software without restriction, including without limitation the rights to
## use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
## of the Software, and to permit persons to whom the Software is furnished to do
## so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
################################################################################

# Shared WSL platform support library (dxg_shared).
# Provides common headers (wddm_types.h, dxcore_loader.h) and the
# DxcoreLoader implementation used by rocdxg, thunk_proxy and amdsmi.

add_library(dxg_shared STATIC
    src/dxcore_loader.cpp
    src/status.cpp
    src/platform.cpp
    src/lda_chain.cpp
    src/device.cpp
    src/utils.cpp
)

set_target_properties(dxg_shared PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(dxg_shared
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<INSTALL_INTERFACE:>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(dxg_shared SYSTEM PRIVATE ${WIN_SDK})

target_compile_options(dxg_shared PRIVATE
    -fPIC
    -Wall
    -Wextra
)

target_compile_definitions(dxg_shared PRIVATE
    LINUX
    __AMD64__
    LITTLEENDIAN_CPU
)

target_link_libraries(dxg_shared PRIVATE ${CMAKE_DL_LIBS})

message(STATUS "dxg_shared configuration:")
message(STATUS "  Include dir: ${CMAKE_CURRENT_SOURCE_DIR}/include")

# Prebuilt thunk_proxy static library.
set(THUNK_PROXY_PREBUILT "${CMAKE_CURRENT_SOURCE_DIR}/src/thunk_proxy/libthunk_proxy.a")

if(NOT EXISTS "${THUNK_PROXY_PREBUILT}")
    message(FATAL_ERROR "Missing prebuilt thunk_proxy archive: ${THUNK_PROXY_PREBUILT}")
endif()

add_library(thunk_proxy STATIC IMPORTED GLOBAL)

set_target_properties(thunk_proxy PROPERTIES
    IMPORTED_LOCATION "${THUNK_PROXY_PREBUILT}"
    INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>;$<INSTALL_INTERFACE:include>"
    INTERFACE_LINK_LIBRARIES dxg_shared
)
