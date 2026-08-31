# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message(
    "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
)
message(
    "                          Cmake amdcuid_example                           "
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

set(AMDCUID_EXAMPLE_EXECUTABLE amdcuid_example)
set(SRC_LIST main.cc)

add_executable(${AMDCUID_EXAMPLE_EXECUTABLE} ${SRC_LIST})

target_include_directories(
    ${AMDCUID_EXAMPLE_EXECUTABLE}
    PRIVATE ${CMAKE_SOURCE_DIR}/lib
)

# Link against the CUID library (public API only)
# The example now uses only the public API defined in amd_cuid.h
target_link_libraries(${AMDCUID_EXAMPLE_EXECUTABLE} amdcuid_static pthread)

# Install example source into documentation
install(
    FILES ${SRC_LIST}
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT ${EXAMPLE_COMPONENT}
)

message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  Finished Cmake amdcuid_example                   ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
