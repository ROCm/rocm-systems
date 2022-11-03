# Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.
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

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                             Cmake rdci                                 ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("-Pkg.-Install Prefix: " ${CPACK_PACKAGING_INSTALL_PREFIX})
message("-CMake inst. Bindir : " ${CMAKE_INSTALL_BINDIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------RSMI Lib Dir: " ${RSMI_LIB_DIR})
message("--------RSMI Inc Dir: " ${RSMI_INC_DIR})
message("-------GRPC ROOT Dir: " ${GRPC_ROOT})
message("")

## Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -m64")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -std=c++11 -pthread ")
# Use this instead of above for 32 bit
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")

if("${CMAKE_BUILD_TYPE}" STREQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -s")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0 -DDEBUG")
endif()

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIB_BOOSTRAP_DIR "${PROJECT_BINARY_DIR}/rdc_libs")

include_directories(${INC_DIR} ${PROJECT_SOURCE_DIR}/include
    "${GRPC_ROOT}/include"
    ${PROJECT_SOURCE_DIR})

set(RDCI_SRC_LIST
    "${COMMON_DIR}/rdc_fields_supported.cc"
    "${COMMON_DIR}/rdc_utils.cc"
    "${SRC_DIR}/RdciDiagSubSystem.cc"
    "${SRC_DIR}/RdciDiscoverySubSystem.cc"
    "${SRC_DIR}/RdciDmonSubSystem.cc"
    "${SRC_DIR}/RdciFieldGroupSubSystem.cc"
    "${SRC_DIR}/RdciGroupSubSystem.cc"
    "${SRC_DIR}/RdciStatsSubSystem.cc"
    "${SRC_DIR}/RdciSubSystem.cc"
    "${SRC_DIR}/rdci.cc")
message("RDCI_SRC_LIST=${RDCI_SRC_LIST}")
set(RDCI_EXE "rdci")

link_directories(${LIB_BOOSTRAP_DIR} "${GRPC_ROOT}/lib" "${GRPC_ROOT}/lib64")

add_executable(${RDCI_EXE} "${RDCI_SRC_LIST}")

## RUNPATH should help to run "rdci" without the usage of ldconfig
if(DEFINED ENV{ROCM_RPATH})
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,--enable-new-dtags -Wl,--rpath,${LIB_RUNPATH} -Wl,--no-as-needed -Wl,-z,origin")
    set(CMAKE_SKIP_BUILD_RPATH TRUE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_SKIP_RPATH TRUE)

    # Set other library
    set(RDCD_EXTRA_LIB ${RDCD_EXTRA_LIB} address_sorting gpr upb ssl crypto
        gRPC::grpc++)
endif()

target_link_libraries(${RDCI_EXE} ${RDCD_EXTRA_LIB} pthread dl rdc_bootstrap)


## Add the install directives for the rdci executable
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${RDCI_EXE}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
    COMPONENT ${CLIENT_COMPONENT}
    DESTINATION ${CMAKE_INSTALL_BINDIR})


message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake rdci                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
