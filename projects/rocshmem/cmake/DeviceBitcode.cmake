###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# Device bitcode for JIT linking: librocshmem_device_{arch}.bc

find_program(LLVM_CLANG clang++
             PATHS
               ${ROCM_PATH}/llvm/bin
               ${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin
             NO_DEFAULT_PATH QUIET)
find_program(LLVM_LINK llvm-link
             PATHS
               ${ROCM_PATH}/llvm/bin
               ${THEROCK_TOOLCHAIN_ROOT}/lib/llvm/bin
               NO_DEFAULT_PATH QUIET)

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

# Convert arch feature suffix to a list of llc -mattr=<feature> flags.
# gfx950:sramecc+:xnack-  →  CMake list: "-mattr=+sramecc;-mattr=-xnack"
# Caller uses the list directly in add_custom_command COMMAND so each element
# becomes a separate shell argument (avoiding comma-joining issues).
function(arch_features_to_mattr_flags full_arch out_var)
  # Split the full arch string on ":" into a list
  string(REPLACE ":" ";" _all_tokens "${full_arch}")
  # Skip the first token (the base arch name) and process the rest as features
  list(LENGTH _all_tokens _ntokens)
  set(_flags "")
  if(_ntokens GREATER 1)
    list(SUBLIST _all_tokens 1 -1 _feat_tokens)
    foreach(_tok ${_feat_tokens})
      if(_tok STREQUAL "")
        continue()
      endif()
      # "sramecc+" → "+sramecc", "xnack-" → "-xnack"
      string(REGEX REPLACE "([a-zA-Z0-9_]+)([+-])$" "\\2\\1" _part "${_tok}")
      list(APPEND _flags "-mattr=${_part}")
    endforeach()
  endif()
  set(${out_var} "${_flags}" PARENT_SCOPE)
endfunction()

# Resolve the default arch list: GPU_TARGETS if set, otherwise auto-detect local GPUs.
if(GPU_TARGETS)
  # Convert comma-separated string to CMake list (semicolon-separated)
  # This handles both -DGPU_TARGETS=gfx942,gfx950 and -DGPU_TARGETS="gfx942;gfx950"
  string(REPLACE "," ";" _GPU_TARGETS_LIST "${GPU_TARGETS}")
  # Ensure it's treated as a list even if already semicolon-separated
  set(_GPU_TARGETS_LIST ${_GPU_TARGETS_LIST})
  strip_arch_features("${_GPU_TARGETS_LIST}" _BITCODE_DEFAULT_ARCHS)
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

# Build a map of base-arch → full target string by querying rocminfo.
# GPU_TARGETS/rocm_local_targets strip feature suffixes (e.g. :sramecc+:xnack-)
# before they reach us, so we re-query rocminfo to recover them.  The result is
# used by downstream cmake (CMakeDeviceBitcodeTester) to pass the correct -mattr
# to llc, which embeds the feature suffix in the HSACO amdhsa.target metadata
# string that HIP checks when loading the module.
# Only query rocminfo when there are architectures to look up — avoids
# an unnecessary subprocess (and potential driver noise) on GPU-less hosts.
find_program(_ROCMINFO rocminfo PATHS ${ROCM_PATH}/bin NO_DEFAULT_PATH QUIET)
if(_ROCMINFO AND _BITCODE_DEFAULT_ARCHS)
  set(_rocminfo_tmpfile "${CMAKE_BINARY_DIR}/CMakeFiles/rocminfo_out.txt")
  execute_process(
    COMMAND ${_ROCMINFO}
    OUTPUT_FILE "${_rocminfo_tmpfile}"
    ERROR_QUIET
  )
  # Parse lines like: "Name:  amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"
  # Use REGEX filter to avoid loading the entire file into a CMake list (which would
  # be split on semicolons in the output and lose entries).
  file(STRINGS "${_rocminfo_tmpfile}" _rocminfo_isa_lines REGEX "amdgcn-amd-amdhsa--")
  foreach(_line ${_rocminfo_isa_lines})
    if(_line MATCHES "Name:.*amdgcn-amd-amdhsa--([a-zA-Z0-9_:+.-]+)")
      set(_full_target "${CMAKE_MATCH_1}")
      string(REGEX REPLACE ":.*" "" _base "${_full_target}")
      # Only record if not already in the map (first GPU wins per arch)
      if(NOT DEFINED _ROCMINFO_ARCH_${_base})
        set(_ROCMINFO_ARCH_${_base} "${_full_target}")
      endif()
    endif()
  endforeach()
endif()

# Build BITCODE_GPU_ARCHS_FULL: for each stripped arch, use rocminfo's full string
# if available, otherwise fall back to the bare arch name.
set(_BITCODE_FULL_LIST "")
foreach(_arch ${_BITCODE_DEFAULT_ARCHS})
  if(DEFINED _ROCMINFO_ARCH_${_arch})
    list(APPEND _BITCODE_FULL_LIST "${_ROCMINFO_ARCH_${_arch}}")
  else()
    list(APPEND _BITCODE_FULL_LIST "${_arch}")
  endif()
endforeach()
set(BITCODE_GPU_ARCHS_FULL "${_BITCODE_FULL_LIST}" CACHE STRING "Full GPU arch strings with features for device bitcode")

# -fvisibility=default ensures extern "C" device API symbols remain
# externally visible after llvm-link and llc.
set(BITCODE_COMPILE_FLAGS_BASE
    -Wall
    -Wextra
    -x hip
    --cuda-device-only
    -std=c++17
    -emit-llvm
    -fvisibility=default
    -O3
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rocshmem_tile_gpu.cpp
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
