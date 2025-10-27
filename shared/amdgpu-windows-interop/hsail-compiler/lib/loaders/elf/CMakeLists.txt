##
########################################################################################################################
#
#  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
#
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_library(clCompilerElfLoader ${sources})

add_subdirectory(utils)

include_directories(${lib})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${elf-toolchain}/common)
if(WIN32)
include_directories(${elf-toolchain}/common/win32)
endif()
include_directories(${elf-toolchain}/libelf)

add_definitions(-DBSD_LIBELF)
