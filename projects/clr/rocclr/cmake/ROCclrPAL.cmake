# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(PAL_CLIENT "OCL")

set(PAL_CLIENT_INTERFACE_MAJOR_VERSION     954)
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
if (AMD_COMPUTE_WIN AND NOT LIB_SRC_BUILD)
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
