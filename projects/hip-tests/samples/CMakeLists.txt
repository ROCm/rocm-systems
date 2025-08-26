
# Copyright (c) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.16.8)
project(hip-samples)

option(ENABLE_ADDRESS_SANITIZER "Option to enable ASAN build" OFF)
option(BUILD_SHARED_LIBS "Use this option to link with shared hip runtime libraries" ON)

if(NOT WIN32)
    set(CPACK_SET_DESTDIR ON CACHE BOOL "Installer package will install hip-samples to CMAKE_INSTALL_PREFIX instead of CPACK_PACKAGING_INSTALL_PREFIX")
endif()
# Read -DROCM_Path and env{ROCM_PATH}
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE STRING "ROCM Path")
    endif()
endif()

# Read -DHIP_Path and env{HIP_PATH}
if(NOT DEFINED HIP_PATH)
    if(DEFINED ENV{HIP_PATH})
        set(HIP_PATH $ENV{HIP_PATH} CACHE STRING "HIP Path")
    endif()
endif()

if(UNIX)
  set(HIPCONFIG_EXEC "hipconfig")
else()
  set(HIPCONFIG_EXEC "hipconfig.bat")
endif()

# both are not set
if(NOT DEFINED HIP_PATH AND NOT DEFINED ROCM_PATH)
    set(HIP_PATH "/opt/rocm")
    set(ROCM_PATH "/opt/rocm")
elseif(DEFINED HIP_PATH AND NOT DEFINED ROCM_PATH)
    execute_process(COMMAND ${HIP_PATH}/bin/${HIPCONFIG_EXEC} --rocmpath
                OUTPUT_VARIABLE ROCM_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)
elseif(DEFINED ROCM_PATH AND NOT DEFINED HIP_PATH)
    set(HIP_PATH ${ROCM_PATH})
endif()
message(STATUS "HIP_PATH: ${HIP_PATH}")
message(STATUS "ROCM_PATH: ${ROCM_PATH}")

execute_process(COMMAND ${HIP_PATH}/bin/${HIPCONFIG_EXEC} --version
                OUTPUT_VARIABLE HIP_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE "." ";" VERSION_LIST ${HIP_VERSION})
list(GET VERSION_LIST 0 HIP_VERSION_MAJOR)
list(GET VERSION_LIST 1 HIP_VERSION_MINOR)
list(GET VERSION_LIST 2 HIP_VERSION_PATCH_GITHASH)
string(REPLACE "-" ";" VERSION_LIST ${HIP_VERSION_PATCH_GITHASH})
list(GET VERSION_LIST 0 HIP_VERSION_PATCH)


if(DEFINED ENV{ROCM_LIBPATCH_VERSION})
   set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}.$ENV{ROCM_LIBPATCH_VERSION})
else()
   set(HIP_PACKAGING_VERSION_PATCH ${HIP_VERSION_PATCH}-${HIP_VERSION_GITHASH})
endif()

#TODO add instructions to build samples at once

set(CMAKE_CXX_EXTENSIONS OFF)

# Address sanitizer options
if(ENABLE_ADDRESS_SANITIZER)
    message(STATUS "Building catch tests with Address Sanitizer options")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan -g -gz")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan -g -gz")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--enable-new-dtags -fuse-ld=lld -fsanitize=address -shared-libasan -g -gz -Wl,--build-id=sha1 -L${ROCM_PATH}/lib/asan")
endif()

add_custom_target(build_samples
            DEPENDS build_intro build_utils build_cookbook
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(0_Intro)
add_subdirectory(1_Utils)
add_subdirectory(2_Cookbook)


# make package_samples to generate packages for samples
set(SAMPLES_SRC ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(packaging)
if(UNIX)
add_custom_target(package_samples COMMAND ${CMAKE_COMMAND} .
    COMMAND rm -rf *.deb *.rpm *.tar.gz
    COMMAND make package
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
add_custom_target(package_samples COMMAND ${CMAKE_COMMAND} .
    COMMAND cpack
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
