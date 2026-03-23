# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Detect unique GPU architectures for add_custom_command
if(NOT DEFINED HIP_ARCH_DETECTION_DONE)
  set(HIP_ARCH_DETECTION_DONE TRUE CACHE INTERNAL "HIP architecture detection completed")

  # Detect GPU architectures for code object generation
  # Note: CMake's HIP language support may not deduplicate architectures
  # Fix any duplicates in CMAKE_HIP_ARCHITECTURES if already set
  if(DEFINED CMAKE_HIP_ARCHITECTURES AND CMAKE_HIP_ARCHITECTURES)
    # Convert to list and remove duplicates
    string(REPLACE ";" ";" GPU_ARCH_LIST "${CMAKE_HIP_ARCHITECTURES}")
    list(REMOVE_ITEM GPU_ARCH_LIST "gfx000" "")
    list(REMOVE_DUPLICATES GPU_ARCH_LIST)
    set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCH_LIST}" CACHE STRING "HIP architectures" FORCE)
    message(STATUS "Deduplicated HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
  elseif(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR NOT CMAKE_HIP_ARCHITECTURES)
    # Auto-detect if not set
    if(NOT DEFINED ROCM_PATH)
      if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
      else()
        set(ROCM_PATH "/opt/rocm")
      endif()
    endif()

    execute_process(
      COMMAND ${ROCM_PATH}/bin/rocm_agent_enumerator
      OUTPUT_VARIABLE DETECTED_GPUS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )

    if(DETECTED_GPUS)
      string(REPLACE "\n" ";" GPU_ARCH_LIST "${DETECTED_GPUS}")
      list(REMOVE_ITEM GPU_ARCH_LIST "gfx000" "")
      list(REMOVE_DUPLICATES GPU_ARCH_LIST)
      set(CMAKE_HIP_ARCHITECTURES "${GPU_ARCH_LIST}" CACHE STRING "HIP architectures" FORCE)
      message(STATUS "Detected HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "HIP architectures" FORCE)
      message(STATUS "Could not detect GPU, using default: ${CMAKE_HIP_ARCHITECTURES}")
    endif()
  endif()
endif()

# For custom commands that need --offload-arch flags, convert the list to multiple flags
# This needs to be regenerated each time in case CMAKE_HIP_ARCHITECTURES changed
set(OFFLOAD_ARCH_FLAGS "")
foreach(arch ${CMAKE_HIP_ARCHITECTURES})
  list(APPEND OFFLOAD_ARCH_FLAGS "--offload-arch=${arch}")
endforeach()

# Build flags for device-only compilation (CMAKE_HIP_FLAGS minus --offload-arch)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _HIP_BUILD_TYPE_UPPER)
separate_arguments(HIP_DEVICE_BUILD_FLAGS_NO_ARCH NATIVE_COMMAND
  "${CMAKE_HIP_FLAGS} ${CMAKE_HIP_FLAGS_${_HIP_BUILD_TYPE_UPPER}}")
list(FILTER HIP_DEVICE_BUILD_FLAGS_NO_ARCH EXCLUDE REGEX "--offload-arch=")
