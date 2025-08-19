# Copyright (c) 2020 - 2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

set(PAL_CLIENT "OCL")

set(PAL_CLIENT_INTERFACE_MAJOR_VERSION     932)
set(GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION 42)
set(GPUOPEN_CLIENT_INTERFACE_MINOR_VERSION 0)
set(AMD_DK_ROOT $ENV{DK_ROOT})

set(PAL_CLOSED_SOURCE       ON)
set(PAL_DEVELOPER_BUILD     OFF)
set(PAL_BUILD_GPUOPEN       ON)
set(PAL_BUILD_SCPC          OFF)
set(PAL_BUILD_VIDEO         OFF)
set(PAL_BUILD_DTIF          OFF)
set(PAL_BUILD_OSS           ON)
set(PAL_BUILD_SECURITY      OFF)
set(PAL_SPPAP_CLOSED_SOURCE OFF)
set(PAL_BUILD_GFX           ON)
set(PAL_BUILD_NULL_DEVICE   OFF)


# The following flags force on certain ASIC support
# This is used for development and test purpose
# Please do not set following flags in staging and mainline in new ASICs
set(PAL_BUILD_GFX6          ON)
set(PAL_BUILD_GFX9          ON)
set(PAL_BUILD_GFX11         ON)
set(PAL_BUILD_NAVI31        ON)
set(PAL_BUILD_NAVI32        ON)
set(PAL_BUILD_NAVI33        ON)
set(PAL_BUILD_PHOENIX1      ON)
# Please do not set above flags in staging and mainline in new ASICs

set(PAL_BRANCHDEFS          ON)
if (AMD_COMPUTE_WIN)
  find_package(AMD_PAL_LIB)
else()
  find_package(AMD_PAL)
endif()

find_package(AMD_HSA_LOADER)

target_sources(rocclr PRIVATE
  ${ROCCLR_SRC_DIR}/device/pal/palappprofile.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palblit.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palconstbuf.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palcounters.cpp
  ${ROCCLR_SRC_DIR}/device/pal/paldevice.cpp
  ${ROCCLR_SRC_DIR}/device/pal/paldeviced3d10.cpp
  ${ROCCLR_SRC_DIR}/device/pal/paldeviced3d11.cpp
  ${ROCCLR_SRC_DIR}/device/pal/paldeviced3d9.cpp
  ${ROCCLR_SRC_DIR}/device/pal/paldevicegl.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palgpuopen.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palkernel.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palmemory.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palprintf.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palprogram.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palresource.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palblitcl.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palsettings.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palsignal.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palthreadtrace.cpp
  ${ROCCLR_SRC_DIR}/device/pal/paltimestamp.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palubercapturemgr.cpp
  ${ROCCLR_SRC_DIR}/device/pal/palvirtual.cpp)

target_compile_definitions(rocclr PUBLIC WITH_PAL_DEVICE PAL_GPUOPEN_OCL)
target_link_libraries(rocclr PUBLIC pal amdhsaloader)

# support for OGL/D3D interop
if(WIN32)
  target_link_libraries(rocclr PUBLIC dxguid.lib)
endif()
