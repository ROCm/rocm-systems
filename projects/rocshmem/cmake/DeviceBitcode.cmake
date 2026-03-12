###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# Device bitcode for JIT linking: librocshmem_device_{arch}.bc

find_program(LLVM_CLANG clang++ PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH QUIET)
find_program(LLVM_LINK llvm-link PATHS ${ROCM_PATH}/llvm/bin NO_DEFAULT_PATH QUIET)

if(NOT LLVM_CLANG OR NOT LLVM_LINK)
  message(WARNING "ROCm LLVM tools (clang++, llvm-link) not found under "
                  "${ROCM_PATH}/llvm/bin; skipping device bitcode targets.")
  return()
endif()

# Derive default bitcode arch list from GPU_TARGETS (strip target features like :xnack-, deduplicate)
set(_BITCODE_DEFAULT_ARCHS "")
foreach(_target ${GPU_TARGETS})
  string(REGEX REPLACE ":.*" "" _base_arch "${_target}")
  list(APPEND _BITCODE_DEFAULT_ARCHS "${_base_arch}")
endforeach()
list(REMOVE_DUPLICATES _BITCODE_DEFAULT_ARCHS)

set(BITCODE_GPU_ARCHS "${_BITCODE_DEFAULT_ARCHS}" CACHE STRING "GPU architectures for device bitcode (semicolon-separated)")

# -fvisibility=default ensures extern "C" device API symbols remain
# externally visible after llvm-link and llc.
set(BITCODE_COMPILE_FLAGS_BASE
    -x hip
    --cuda-device-only
    -std=c++20
    -emit-llvm
    -fvisibility=default
    -I${CMAKE_CURRENT_SOURCE_DIR}/include/rocshmem
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${CMAKE_BINARY_DIR}/include
    -I${CMAKE_BINARY_DIR}/include/rocshmem
)

# Add MPI include directories — rocshmem_config.h defines HAVE_EXTERNAL_MPI
# when MPI is found, causing rocshmem_mpi.hpp to #include <mpi.h> transitively.
if(MPI_CXX_FOUND)
  foreach(mpi_include_dir ${MPI_CXX_INCLUDE_DIRS})
    list(APPEND BITCODE_COMPILE_FLAGS_BASE -I${mpi_include_dir})
  endforeach()
endif()

# Core device sources (backend_bc.cpp is host-only; backend_bc_device.cpp provides
# the device-side create_ctx/destroy_ctx dispatchers)
set(BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rocshmem_gpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc_policy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/team.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sync/abql_block_mutex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/context_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend_bc_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/device/rocshmem_wrapper.cc
)

# Backend-specific device sources
if(USE_RO)
  list(APPEND BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reverse_offload/backend_ro.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reverse_offload/context_ro_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reverse_offload/queue.cpp
  )
endif()

if(USE_IPC)
  list(APPEND BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/backend_ipc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/context_ipc_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ipc/context_ipc_device_coll.cpp
  )
endif()

if(USE_GDA)
  list(APPEND BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/context_gda_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/context_gda_device_coll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/backend_gda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/queue_pair.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/ionic/queue_pair_ionic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/mlx5/queue_pair_mlx5.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/bnxt/queue_pair_bnxt.cpp
  )
endif()

# Build bitcode for each GPU architecture
set(ALL_BITCODE_OUTPUTS)
foreach(gpu_arch ${BITCODE_GPU_ARCHS})
  set(BITCODE_COMPILE_FLAGS ${BITCODE_COMPILE_FLAGS_BASE} --offload-arch=${gpu_arch})
  set(BITCODE_OBJECTS_${gpu_arch})
  foreach(src_file ${BITCODE_SOURCES})
    get_filename_component(src_name ${src_file} NAME_WE)
    set(bc_file ${CMAKE_CURRENT_BINARY_DIR}/bitcode/${gpu_arch}/${src_name}.bc)
    list(APPEND BITCODE_OBJECTS_${gpu_arch} ${bc_file})

    add_custom_command(
      OUTPUT ${bc_file}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/bitcode/${gpu_arch}
      COMMAND ${LLVM_CLANG} ${BITCODE_COMPILE_FLAGS} -c ${src_file} -o ${bc_file}
      DEPENDS ${src_file}
      COMMENT "Compiling ${src_name} to bitcode for ${gpu_arch}"
      VERBATIM
    )
  endforeach()

  set(BITCODE_OUTPUT_${gpu_arch} ${CMAKE_CURRENT_BINARY_DIR}/librocshmem_device_${gpu_arch}.bc)
  list(APPEND ALL_BITCODE_OUTPUTS ${BITCODE_OUTPUT_${gpu_arch}})

  add_custom_command(
    OUTPUT ${BITCODE_OUTPUT_${gpu_arch}}
    COMMAND ${LLVM_LINK} ${BITCODE_OBJECTS_${gpu_arch}} -o ${BITCODE_OUTPUT_${gpu_arch}}
    DEPENDS ${BITCODE_OBJECTS_${gpu_arch}}
    COMMENT "Linking device bitcode for ${gpu_arch}"
    VERBATIM
  )

  install(
    FILES ${BITCODE_OUTPUT_${gpu_arch}}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT runtime
  )

  message(STATUS "Device bitcode for ${gpu_arch}: ${BITCODE_OUTPUT_${gpu_arch}}")
endforeach()

add_custom_target(rocshmem_device_bitcode ALL
  DEPENDS ${ALL_BITCODE_OUTPUTS}
)

message(STATUS "Device bitcode will be built for architectures: ${BITCODE_GPU_ARCHS}")
