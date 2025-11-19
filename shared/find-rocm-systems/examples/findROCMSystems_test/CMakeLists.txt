################################################################################
# Example Test CMAKE Using find ROCm Systems
# 
# This example demonstrates how to use find_package(rocm-systems)
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
################################################################################

cmake_minimum_required(VERSION 3.18)
project(FindROCmSystemsTestApp VERSION 1.0.0)

################################################################################
# Find ROCm Systems
################################################################################
find_package(rocm-systems REQUIRED)
if(rocm-systems_FOUND)
  message(STATUS "========================================")
  message(STATUS "ROCm Systems Test Application")
  message(STATUS "========================================")
  message(STATUS "ROCm Systems Version: ${ROCM_SYSTEMS_VERSION}")
  message(STATUS "Installation Prefix:  ${ROCM_SYSTEMS_PREFIX}")
  message(STATUS "Library Directory:    ${ROCM_SYSTEMS_LIB_DIR}")
  message(STATUS "Include Directory:    ${ROCM_SYSTEMS_INCLUDE_DIR}")
  message(STATUS "Binary Directory:     ${ROCM_SYSTEMS_BIN_DIR}")
  message(STATUS "")
  message(STATUS "Available Components: ${ROCM_SYSTEMS_AVAILABLE_COMPONENTS}")
  message(STATUS "Found Components:     ${rocm-systems_FOUND_COMPONENTS}")
  message(STATUS "")
  message(STATUS "Component Status:")
  message(STATUS "  rdc:                ${rocm-systems_rdc_FOUND}")
  message(STATUS "  rocm-core:          ${rocm-systems_rocm-core_FOUND}")
  message(STATUS "  rocm_smi:           ${rocm-systems_rocm_smi_FOUND}")
  message(STATUS "  rocprofiler-register:${rocm-systems_rocprofiler-register_FOUND}")
  message(STATUS "  rocprofiler-sdk:    ${rocm-systems_rocprofiler-sdk_FOUND}")
  message(STATUS "  rocprofiler-systems:${rocm-systems_rocprofiler-systems_FOUND}")
  message(STATUS "  rocr-runtime:       ${rocm-systems_rocr-runtime_FOUND}")
  message(STATUS "  hip:                ${rocm-systems_hip_FOUND}")
  message(STATUS "========================================")
endif()


