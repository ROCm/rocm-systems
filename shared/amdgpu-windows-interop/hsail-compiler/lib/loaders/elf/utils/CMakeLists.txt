##
########################################################################################################################
#
#  Copyright (C) 2020 Advanced Micro Devices, Inc.  All rights reserved.
#
########################################################################################################################

cmake_minimum_required(VERSION 3.10)

if(NOT TARGET oclelf)
  add_subdirectory(libelf)
endif()
