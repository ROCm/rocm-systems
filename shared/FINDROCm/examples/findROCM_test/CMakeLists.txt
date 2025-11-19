################################################################################
# Example Test CMAKE Using find FINDROCm CMAKE Config
# 
# This example demonstrates how to use find_package(FINDROCm)
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
################################################################################

cmake_minimum_required(VERSION 3.18)
project(FINDROCmTestApp VERSION 1.0.0)

################################################################################
# Find ROCm
################################################################################
find_package(FINDROCm REQUIRED)
if(FINDROCm_FOUND)
  message(STATUS "========================================")
  message(STATUS "FINDROCm Test Application")
  message(STATUS "========================================")
  message(STATUS "FINDROCm Version: ${FINDROCM_VERSION}")
  message(STATUS "Installation Prefix:  ${FINDROCM_PREFIX}")
  message(STATUS "Library Directory:    ${FINDROCM_LIB_DIR}")
  message(STATUS "Include Directory:    ${FINDROCM_INCLUDE_DIR}")
  message(STATUS "Binary Directory:     ${FINDROCM_BIN_DIR}")
  message(STATUS "")
  message(STATUS "Available Components: ${FINDROCM_AVAILABLE_COMPONENTS}")
  message(STATUS "Found Components:     ${FINDROCm_FOUND_COMPONENTS}")
  message(STATUS "")
  message(STATUS "Component Status:")
  foreach(COMP IN LISTS FINDROCm_FOUND_COMPONENTS)
	  message(STATUS "${COMP}:${FINDROCm_${COMP}_FOUND}")
  endforeach()
  message(STATUS "========================================")

endif()


