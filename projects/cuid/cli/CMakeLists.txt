# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message(
    "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
)
message(
    "                            Cmake amdcuid_tool                            "
)
message(
    "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
)

message("")
message("Build Configuration:")
message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("------Install Prefix: " ${CMAKE_INSTALL_PREFIX})
message("-CMake inst. Bindir : " ${CMAKE_INSTALL_BINDIR})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("")

set(INC_DIR ${CMAKE_SOURCE_DIR}/lib)
set(AMDCUID_TOOL_EXECUTABLE amdcuid_tool)
set(SRC_LIST amdcuid_tool.cc)

# Add the tool executable
add_executable(${AMDCUID_TOOL_EXECUTABLE} ${SRC_LIST})

# Include directories - only need access to public headers
target_include_directories(${AMDCUID_TOOL_EXECUTABLE} PRIVATE ${INC_DIR})

# Link against the CUID library (public API only)
# The tool now uses only the public API defined in amd_cuid.h
target_link_libraries(${AMDCUID_TOOL_EXECUTABLE} amdcuid_static OpenSSL::Crypto)

# Install the tool
install(
    TARGETS ${AMDCUID_TOOL_EXECUTABLE}
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${CLI_COMPONENT}
)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                    Finished Cmake amdcuid_tool                    ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
