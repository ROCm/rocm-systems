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
message("                             Cmake rdci                                   ")
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
message("--------SMI Lib Dir: " ${AMD_SMI_LIB_DIR})
message("--------SMI Inc Dir: " ${AMD_SMI_INCLUDE_DIR})
message("-------GRPC ROOT Dir: " ${GRPC_ROOT})
message("")

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${INC_DIR} ${PROJECT_SOURCE_DIR}/include "${GRPC_ROOT}/include"
                    ${PROJECT_SOURCE_DIR} ${AMD_SMI_INCLUDE_DIR})

set(RDCI_SRC_LIST
    "${COMMON_DIR}/rdc_fields_supported.cc"
    "${COMMON_DIR}/rdc_utils.cc"
    "${SRC_DIR}/RdciDiagSubSystem.cc"
    "${SRC_DIR}/RdciDiscoverySubSystem.cc"
    "${SRC_DIR}/RdciDmonSubSystem.cc"
    "${SRC_DIR}/RdciFieldGroupSubSystem.cc"
    "${SRC_DIR}/RdciGroupSubSystem.cc"
    "${SRC_DIR}/RdciStatsSubSystem.cc"
    "${SRC_DIR}/RdciPolicySubSystem.cc"
    "${SRC_DIR}/RdciHealthSubSystem.cc"
    "${SRC_DIR}/RdciConfigSubSystem.cc"
    "${SRC_DIR}/RdciSubSystem.cc"
    "${SRC_DIR}/RdciTopologyLinkSubSystem.cc"
    "${SRC_DIR}/RdciXgmiLinkStatusSubSystem.cc"
    "${SRC_DIR}/rdci.cc")
message("RDCI_SRC_LIST=${RDCI_SRC_LIST}")
set(RDCI_EXE "rdci")

link_directories(${AMD_SMI_LIB_DIR})

add_executable(${RDCI_EXE} "${RDCI_SRC_LIST}")
target_compile_definitions(${RDCI_EXE} PRIVATE CURRENT_GIT_HASH=${GIT_HASH})

# help find librdc_bootstrap.so at runtime
set_target_properties(${RDCI_EXE} PROPERTIES INSTALL_RPATH
                                             "\$ORIGIN/../lib:\$ORIGIN/../lib/rdc/grpc/lib")

target_link_libraries(${RDCI_EXE} pthread dl gRPC::grpc++ rdc_bootstrap)

## Add the install directives for the rdci executable
install(
    TARGETS ${RDCI_EXE}
    PERMISSIONS
        OWNER_READ
        OWNER_WRITE
        OWNER_EXECUTE
        GROUP_READ
        GROUP_EXECUTE
        WORLD_READ
        WORLD_EXECUTE
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${CLIENT_COMPONENT})

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake rdci                          ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
