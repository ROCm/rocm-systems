##
########################################################################################################################
#
#  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
#
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
if(WIN32)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/win32)
else()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/win32/sys)
endif()

add_definitions(-DBSD_LIBELF -DUSE_MEMFILE)

file(GLOB sources
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/*.c
  ${CMAKE_CURRENT_SOURCE_DIR}/*.h
)
add_library(oclelf STATIC ${sources})
set_target_properties(oclelf PROPERTIES POSITION_INDEPENDENT_CODE ON)

