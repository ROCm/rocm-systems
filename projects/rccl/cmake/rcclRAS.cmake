# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

cmake_minimum_required(VERSION 3.16)

message("Building rccl RAS client executable")

# NOTE:
# The CMake migration work in this workspace can change when/if the hipify "src"
# tree exists at configure time. Referencing a hipified path here makes CMake
# fail early if that generated file isn't registered yet.
#
# For robustness, build the RAS client from the real source file.
add_executable(rcclras "${CMAKE_SOURCE_DIR}/src/ras/client.cc")

target_include_directories(rcclras PRIVATE ${PROJECT_BINARY_DIR}/include)
target_include_directories(rcclras PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_include_directories(rcclras PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

target_link_libraries(rcclras PRIVATE hip::host)
target_link_libraries(rcclras PRIVATE dl)

if(BUILD_SHARED_LIBS)
  target_link_libraries(rcclras PRIVATE rccl hip::device)
else()
  add_dependencies(rcclras rccl)
  target_link_libraries(rcclras PRIVATE dl rt -lrccl -L${CMAKE_BINARY_DIR} -lamdhip64 -L${ROCM_PATH}/lib)
endif()


rocm_install(TARGETS rcclras)
