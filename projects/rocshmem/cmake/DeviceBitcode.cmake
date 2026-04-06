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

# Strip feature suffixes (gfx942:sramecc+:xnack- → gfx942) and deduplicate.
function(strip_arch_features targets_list out_var)
  set(_result "")
  foreach(_t ${targets_list})
    string(REGEX REPLACE ":.*" "" _base "${_t}")
    list(APPEND _result "${_base}")
  endforeach()
  list(REMOVE_DUPLICATES _result)
  set(${out_var} "${_result}" PARENT_SCOPE)
endfunction()

# Resolve the default arch list: GPU_TARGETS if set, otherwise auto-detect local GPUs.
if(GPU_TARGETS)
  strip_arch_features("${GPU_TARGETS}" _BITCODE_DEFAULT_ARCHS)
elseif(COMMAND rocm_local_targets)
  rocm_local_targets(_LOCAL_GPUS)
  if(_LOCAL_GPUS)
    strip_arch_features("${_LOCAL_GPUS}" _BITCODE_DEFAULT_ARCHS)
    message(STATUS "GPU_TARGETS not set; auto-detected local GPU(s) for device bitcode: ${_BITCODE_DEFAULT_ARCHS}")
  else()
    message(WARNING "GPU_TARGETS not set and no local GPU detected. "
      "Device bitcode will not be built. Set -DGPU_TARGETS=<arch> to enable.")
  endif()
endif()

set(BITCODE_GPU_ARCHS "${_BITCODE_DEFAULT_ARCHS}" CACHE STRING "GPU architectures for device bitcode (semicolon-separated)")

# -fvisibility=default ensures extern "C" device API symbols remain
# externally visible after llvm-link and llc.
set(BITCODE_COMPILE_FLAGS_BASE
    -x hip
    --cuda-device-only
    -std=c++17
    -emit-llvm
    -fvisibility=default
    -Xclang -mcode-object-version=none
    -I${CMAKE_CURRENT_SOURCE_DIR}/include/rocshmem
    -I${CMAKE_CURRENT_SOURCE_DIR}/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    -I${CMAKE_BINARY_DIR}/include
    -I${CMAKE_BINARY_DIR}/include/rocshmem
)

if(${ROCM_MAJOR_VERSION} LESS 7)
  # ROCm 6.x requires us to explicitly enable warp sync builtins
  list(APPEND BITCODE_COMPILE_FLAGS_BASE -DHIP_ENABLE_WARP_SYNC_BUILTINS=1)
endif()

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

# Backend-specific device sources. The bitcode MUST match the host library's
# backend selection because:
#   1. DISPATCH macros in backend_type.hpp produce different code (switch vs
#      direct static_cast) depending on which USE_* defines are active.
#   2. Context struct layouts differ per backend — static_cast reinterprets the
#      same pointer as different derived types, so ABI must match.
# A "universal" bitcode with all backends forced on would crash when paired
# with a host library compiled for a single backend (layout/ABI mismatch).
# TODO: refactor DISPATCH to remove this hard limitation

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

# GDA queue_pair implementations are guarded by GDA_MLX5/GDA_IONIC/GDA_BNXT in
# queue_pair.hpp. Only compile the backend(s) enabled for this build so that
# declarations and definitions match.
if(USE_GDA)
  list(APPEND BITCODE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/context_gda_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/context_gda_device_coll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/backend_gda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/queue_pair.cpp
  )
  if(GDA_MLX5)
    list(APPEND BITCODE_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/mlx5/queue_pair_mlx5.cpp
    )
  endif()
  if(GDA_IONIC)
    list(APPEND BITCODE_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/ionic/queue_pair_ionic.cpp
    )
  endif()
  if(GDA_BNXT)
    list(APPEND BITCODE_SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/src/gda/bnxt/queue_pair_bnxt.cpp
    )
  endif()
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
