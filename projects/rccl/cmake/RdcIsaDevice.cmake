# cmake/RdcIsaDevice.cmake
#
# Per-arch device compilation pipeline with -fgpu-rdc-isa.
#
# For each GPU target: compile specialized device files + common.cu.cpp with
# --offload-device-only, device-link with ld.lld, then bundle all arch ELFs
# into a fat binary embedded into a host object via --offload-host-only.
#
# This achieves the devicelinker's scheduling model (per-arch parallelism,
# arch-guard filtering, no redundant multi-arch compilation) while keeping
# -fgpu-rdc-isa's simplicity (no assembly extraction or Python driver).
#
# Required variables (set by src/CMakeLists.txt before including):
#   HIPIFY_DIR, GEN_DIR, GPU_TARGETS, PROJECT_BINARY_DIR, PROJECT_SOURCE_DIR,
#   LL128_ENABLED, ENABLE_COMPRESS
#
# Outputs:                                                               jkui
#   RDC_ISA_DEVICE_OBJECTS - object files to link into the rccl target

message(STATUS "RdcIsaDevice: per-arch device compilation pipeline enabled")

# ---------------------------------------------------------------------------
# Find tools (relative to the CXX compiler)
# ---------------------------------------------------------------------------
get_filename_component(_rdi_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)

# Prefer ld.lld next to the CXX compiler — this must match the clang that
# emits the AMDGPU relocations (e.g. R_AMDGPU_LDS_OFFSET handled by our
# custom postRelocatePass).  Falling back to the system ld.lld will cause
# "unknown relocation" errors during device linking.
if(EXISTS "${_rdi_compiler_dir}/ld.lld")
  set(RDI_LLD "${_rdi_compiler_dir}/ld.lld")
else()
  find_program(RDI_LLD NAMES ld.lld
    HINTS "${_rdi_compiler_dir}" "${ROCM_PATH}/llvm/bin"
    NO_DEFAULT_PATH)
  if(NOT RDI_LLD)
    find_program(RDI_LLD NAMES ld.lld REQUIRED)
  endif()
endif()

find_program(RDI_BUNDLER NAMES clang-offload-bundler
  HINTS "${_rdi_compiler_dir}" "${_rdi_compiler_dir}/../lib/llvm/bin"
        "${ROCM_PATH}/llvm/bin" REQUIRED)
set(RDI_CLANG "${CMAKE_CXX_COMPILER}")

message(STATUS "RdcIsaDevice: clang   = ${RDI_CLANG}")
message(STATUS "RdcIsaDevice: lld     = ${RDI_LLD}")
message(STATUS "RdcIsaDevice: bundler = ${RDI_BUNDLER}")

# Extract --hip-path / --hip-device-lib-path from CMAKE_CXX_FLAGS (TheRock)
# Used by the OBJECT library compile (CMake already forwards the rest of
# CMAKE_CXX_FLAGS automatically).
set(RDI_HIP_COMPILER_FLAGS "")
string(REGEX MATCHALL "--hip-path=[^ ]+" _hip_path_flags "${CMAKE_CXX_FLAGS}")
list(APPEND RDI_HIP_COMPILER_FLAGS ${_hip_path_flags})
string(REGEX MATCHALL "--hip-device-lib-path=[^ ]+" _hip_devlib_flags "${CMAKE_CXX_FLAGS}")
list(APPEND RDI_HIP_COMPILER_FLAGS ${_hip_devlib_flags})
if(RDI_HIP_COMPILER_FLAGS)
  message(STATUS "RdcIsaDevice: forwarding HIP flags: ${RDI_HIP_COMPILER_FLAGS}")
endif()


# ---------------------------------------------------------------------------
# Parse GPU_TARGETS: strip target features, build offload-arch flag list
# ---------------------------------------------------------------------------
set(RDI_GPU_TARGETS "")
set(RDI_OFFLOAD_ARCH_FLAGS "")
foreach(_gpu_raw ${GPU_TARGETS})
  string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
  list(APPEND RDI_GPU_TARGETS "${_gpu}")
  list(APPEND RDI_OFFLOAD_ARCH_FLAGS "--offload-arch=${_gpu}")
endforeach()
message(STATUS "RdcIsaDevice: GPU targets = ${RDI_GPU_TARGETS}")

# ---------------------------------------------------------------------------
# Optimization flags
# ---------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(RDI_OPT_FLAGS -O0 -g)
else()
  set(RDI_OPT_FLAGS -O3)
endif()

# ---------------------------------------------------------------------------
# INTERFACE library: shared definitions and includes for device compilation.
# Reads from the rccl target (already fully configured) and directory scope.
# ---------------------------------------------------------------------------
add_library(rccl_rdi_defs INTERFACE)

get_target_property(_rccl_defs rccl COMPILE_DEFINITIONS)
if(_rccl_defs)
  # The per-arch device pipeline compiles spec_*.cpp + common.cu.cpp which
  # DO need ncclDevFuncTable_*[] (dispatched from ncclKernelMain's else-branch).
  # RCCL_DEVICE_TABLE_OMIT is only appropriate for the main rccl target; filter
  # it out here so per-arch OBJECT libraries keep the tables.  Same for
  # RCCL_NO_EXTERN_SHMEM: device-only per-arch compiles rely on extern __shared__
  # which ld.lld resolves via postRelocatePass.
  list(REMOVE_ITEM _rccl_defs RCCL_DEVICE_TABLE_OMIT RCCL_NO_EXTERN_SHMEM)
  target_compile_definitions(rccl_rdi_defs INTERFACE ${_rccl_defs})
endif()

get_directory_property(_dir_defs COMPILE_DEFINITIONS)
if(_dir_defs)
  target_compile_definitions(rccl_rdi_defs INTERFACE ${_dir_defs})
endif()

target_compile_definitions(rccl_rdi_defs INTERFACE
  __HIP_PLATFORM_AMD__=1
  FMT_HEADER_ONLY=1
)

get_target_property(_rccl_includes rccl INCLUDE_DIRECTORIES)
if(_rccl_includes)
  target_include_directories(rccl_rdi_defs INTERFACE ${_rccl_includes})
endif()

set(_hip_includes "")
foreach(_hip_tgt hip::device hip::amdhip64 hip::host)
  if(TARGET ${_hip_tgt} AND NOT _hip_includes)
    get_target_property(_hip_includes ${_hip_tgt} INTERFACE_INCLUDE_DIRECTORIES)
  endif()
endforeach()
if(_hip_includes)
  target_include_directories(rccl_rdi_defs SYSTEM INTERFACE ${_hip_includes})
elseif(ROCM_PATH)
  target_include_directories(rccl_rdi_defs SYSTEM INTERFACE "${ROCM_PATH}/include")
endif()

if(fmt_SOURCE_DIR)
  target_include_directories(rccl_rdi_defs SYSTEM INTERFACE "${fmt_SOURCE_DIR}/include")
elseif(TARGET fmt::fmt-header-only)
  get_target_property(_fmt_inc fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
  if(_fmt_inc)
    foreach(_p ${_fmt_inc})
      if(NOT _p MATCHES "^\\$<")
        target_include_directories(rccl_rdi_defs SYSTEM INTERFACE "${_p}")
      endif()
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------------------
# Read specialized_files.txt manifest (generated by generate.py)
# ---------------------------------------------------------------------------
set(SPECIALIZED_FILES_TXT "${GEN_DIR}/specialized_files.txt")
if(NOT EXISTS "${SPECIALIZED_FILES_TXT}")
  message(FATAL_ERROR "RdcIsaDevice: ${SPECIALIZED_FILES_TXT} not found. "
    "Run generate.py with SPECIALIZED flag first.")
endif()

file(STRINGS "${SPECIALIZED_FILES_TXT}" RDI_ENTRIES)
list(LENGTH RDI_ENTRIES RDI_KERNEL_COUNT)
message(STATUS "RdcIsaDevice: ${RDI_KERNEL_COUNT} specialized functions in manifest")

# ---------------------------------------------------------------------------
# Guard evaluation: skip functions whose #if guard excludes a GPU target.
# ---------------------------------------------------------------------------
function(rdi_evaluate_guard GUARD GPU_TARGET RESULT_VAR)
  if("${GUARD}" STREQUAL "")
    set(${RESULT_VAR} TRUE PARENT_SCOPE)
    return()
  endif()
  if("${GUARD}" MATCHES "ENABLE_LL128" AND NOT LL128_ENABLED)
    set(${RESULT_VAR} FALSE PARENT_SCOPE)
    return()
  endif()
  string(REGEX MATCHALL "__gfx[0-9a-z]+__" _guard_archs "${GUARD}")
  if(NOT _guard_archs)
    set(${RESULT_VAR} TRUE PARENT_SCOPE)
    return()
  endif()
  foreach(_ga ${_guard_archs})
    string(REGEX REPLACE "^__(.+)__$" "\\1" _arch "${_ga}")
    if("${_arch}" STREQUAL "${GPU_TARGET}")
      set(${RESULT_VAR} TRUE PARENT_SCOPE)
      return()
    endif()
  endforeach()
  set(${RESULT_VAR} FALSE PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# Host triple for the offload bundler
# ---------------------------------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _rdi_sys_name)
if(NOT _rdi_sys_name)
  set(_rdi_sys_name "linux")
endif()
set(_rdi_host_triple "${CMAKE_SYSTEM_PROCESSOR}-unknown-${_rdi_sys_name}-gnu")

# ---------------------------------------------------------------------------
# Build include/define flag lists for custom commands
# (OBJECT libraries inherit via target_link_libraries; custom commands need explicit flags)
# ---------------------------------------------------------------------------
set(_rdi_def_flags "")
get_target_property(_defs rccl_rdi_defs INTERFACE_COMPILE_DEFINITIONS)
if(_defs)
  foreach(_d ${_defs})
    list(APPEND _rdi_def_flags "-D${_d}")
  endforeach()
endif()

set(_rdi_inc_flags "")
get_target_property(_incs rccl_rdi_defs INTERFACE_INCLUDE_DIRECTORIES)
if(_incs)
  foreach(_inc ${_incs})
    list(APPEND _rdi_inc_flags "-I${_inc}")
  endforeach()
endif()
get_target_property(_sys_incs rccl_rdi_defs INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
if(_sys_incs)
  foreach(_inc ${_sys_incs})
    list(APPEND _rdi_inc_flags "-isystem${_inc}")
  endforeach()
endif()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
set(RDI_BUILD_DIR "${PROJECT_BINARY_DIR}/device_rdc_isa")
set(SPECIALIZED_DIR "${GEN_DIR}/specialized")

# ===========================================================================
# Per-GPU-target: OBJECT library (device compile) + device link
# ===========================================================================
set(ALL_DEVICE_ELFS "")
set(RDI_BUNDLER_TARGETS "host-${_rdi_host_triple}-")
set(RDI_BUNDLER_INPUTS "--input=/dev/null")

foreach(RDI_GPU_TARGET ${RDI_GPU_TARGETS})
  set(RDI_ARCH_DIR "${RDI_BUILD_DIR}/${RDI_GPU_TARGET}")
  file(MAKE_DIRECTORY ${RDI_ARCH_DIR})

  # =========================================================================
  # Filter specialized sources for this arch
  # =========================================================================
  set(ARCH_SOURCES "")
  set(_rdi_skipped 0)

  foreach(ENTRY ${RDI_ENTRIES})
    if(NOT ENTRY MATCHES "^([^ ]+) +([^ ]+) *(.*)")
      continue()
    endif()
    set(CPP_FILE "${CMAKE_MATCH_1}")
    set(_entry_guard "${CMAKE_MATCH_3}")

    rdi_evaluate_guard("${_entry_guard}" "${RDI_GPU_TARGET}" _guard_ok)
    if(NOT _guard_ok)
      math(EXPR _rdi_skipped "${_rdi_skipped} + 1")
      continue()
    endif()

    list(APPEND ARCH_SOURCES "${SPECIALIZED_DIR}/${CPP_FILE}")
  endforeach()

  list(LENGTH ARCH_SOURCES _rdi_built)
  message(STATUS "RdcIsaDevice [${RDI_GPU_TARGET}]: ${_rdi_built} functions, ${_rdi_skipped} skipped (arch guard)")

  # =========================================================================
  # OBJECT library: per-arch device-only compilation of specialized files
  # AND common.cu.cpp (which contains the generic kernel + dispatch tables).
  #
  # Using an OBJECT library (not custom commands) ensures CMake adds the
  # standard C++ implicit include paths (libstdc++, libc headers) that clang
  # would otherwise miss when invoked directly.
  #
  # The -x hip flag tells clang to parse as HIP; --offload-device-only
  # produces bare AMDGPU ELF objects for a single --offload-arch.
  # =========================================================================
  set(_dev_target "rccl_rdi_${RDI_GPU_TARGET}")

  add_library(${_dev_target} OBJECT
    ${ARCH_SOURCES}
    ${HIPIFY_DIR}/src/device/common.cu.cpp
  )
  target_link_libraries(${_dev_target} PRIVATE rccl_rdi_defs)

  target_compile_options(${_dev_target} PRIVATE
    -x hip
    --offload-device-only
    --offload-arch=${RDI_GPU_TARGET}
    -fgpu-rdc-isa
    ${RDI_OPT_FLAGS}
    -std=c++17
    -w
    ${RDI_HIP_COMPILER_FLAGS}
  )

  add_dependencies(${_dev_target} hipify_all)

  # =========================================================================
  # Device link: ld.lld produces a shared device ELF.
  # postRelocatePass patches kernel descriptors (VGPR, LDS, ACCUM_OFFSET, etc.)
  # =========================================================================
  set(DEVICE_ELF "${RDI_ARCH_DIR}/device.elf")
  add_custom_command(
    OUTPUT  ${DEVICE_ELF}
    COMMAND ${RDI_LLD}
      -shared
      -o ${DEVICE_ELF}
      $<TARGET_OBJECTS:${_dev_target}>
      -Bsymbolic
      --gc-sections
    DEPENDS ${_dev_target}
    COMMENT "RdcIsaDevice [${RDI_GPU_TARGET}] device link -> device.elf"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )

  list(APPEND ALL_DEVICE_ELFS "${DEVICE_ELF}")
  list(APPEND RDI_BUNDLER_TARGETS "hip-amdgcn-amd-amdhsa--${RDI_GPU_TARGET}")
  list(APPEND RDI_BUNDLER_INPUTS "--input=${DEVICE_ELF}")

endforeach()  # end per-GPU-target loop

# ===========================================================================
# Bundle all per-arch device.elf files into a single .hipfb fat binary
# ===========================================================================
set(RDI_HIPFB "${RDI_BUILD_DIR}/device.hipfb")
list(JOIN RDI_BUNDLER_TARGETS "," _rdi_bundler_targets_str)

set(RDI_BUNDLER_COMPRESS "")
if(ENABLE_COMPRESS)
  set(RDI_BUNDLER_COMPRESS "--compress")
endif()

add_custom_command(
  OUTPUT  ${RDI_HIPFB}
  COMMAND ${RDI_BUNDLER}
    --type=bc
    --targets=${_rdi_bundler_targets_str}
    ${RDI_BUNDLER_INPUTS}
    --output=${RDI_HIPFB}
    ${RDI_BUNDLER_COMPRESS}
  DEPENDS ${ALL_DEVICE_ELFS}
  COMMENT "RdcIsaDevice: bundle device ELFs -> device.hipfb [${RDI_GPU_TARGETS}]"
  VERBATIM
)

# ===========================================================================
# Host compile common.cu.cpp with embedded device binary
#
# Uses an OBJECT library (not a custom command) so CMake adds the standard C++
# implicit include paths. The fat binary produced by the bundler is embedded
# via -fcuda-include-gpubinary; the OBJECT library depends on the bundler
# through OBJECT_DEPENDS so the hipfb is materialized first.
# ===========================================================================
set(RDI_HOST_COMPRESS "")
if(ENABLE_COMPRESS)
  set(RDI_HOST_COMPRESS "--offload-compress")
endif()

# common.cu.cpp is used directly (it has been excluded from HIP_SOURCES when
# RDC_ISA_PER_ARCH is on, so there's no double-compile conflict).
add_library(rccl_rdi_host OBJECT ${HIPIFY_DIR}/src/device/common.cu.cpp)
target_link_libraries(rccl_rdi_host PRIVATE rccl_rdi_defs)

# RCCL_NO_EXTERN_SHMEM is a matched pair of guards in common.h / common.cu:
#   - in common.h: emits plain (non-extern) __shared__ declarations, satisfying
#     the host parser (newer clang rejects `extern __shared__` at file scope
#     in --offload-host-only).  Host never reads these; device bodies are
#     skipped in host-only mode.
#   - in common.cu: skips its own __shared__ definition so common.h's copy is
#     the sole definition in this TU (no redefinition conflict).
target_compile_definitions(rccl_rdi_host PRIVATE RCCL_NO_EXTERN_SHMEM)

target_compile_options(rccl_rdi_host PRIVATE
  -x hip
  --offload-host-only
  ${RDI_OFFLOAD_ARCH_FLAGS}
  # SHELL: prefix prevents CMake from deduplicating the repeated -Xclang
  # tokens, which would strip the second one and cause the hipfb path to be
  # treated as an input source file.
  "SHELL:-Xclang -fcuda-include-gpubinary -Xclang ${RDI_HIPFB}"
  ${RDI_OPT_FLAGS}
  -std=c++17
  -fPIC
  -w
  ${RDI_HIP_COMPILER_FLAGS}
  ${RDI_HOST_COMPRESS}
)

# Wire the hipfb dependency at the target level (not source level — which
# would create a cycle because common.cu.cpp is also compiled by the per-arch
# OBJECT libraries that the hipfb transitively depends on).
add_custom_target(rccl_rdi_hipfb DEPENDS ${RDI_HIPFB})
add_dependencies(rccl_rdi_host rccl_rdi_hipfb hipify_all)

# ===========================================================================
# Aux TUs that need dual-pass (host+device) HIP compilation, multi-arch,
# but NOT RDC (each TU self-contained for device).  These files contain the
# few __global__ kernels / <<<>>> launches that live outside the specialized
# device pipeline and therefore cannot go through --offload-host-only on the
# main rccl target.
#   - collectives.cc     : launches hierarchicalAGShuffle via <<<>>>
#   - onerank.cu.cpp     : defines oneRankReduce __global__ kernel
# ===========================================================================
add_library(rccl_rdi_aux OBJECT
  ${HIPIFY_DIR}/src/collectives.cc
  ${HIPIFY_DIR}/src/device/onerank.cu.cpp
)
target_link_libraries(rccl_rdi_aux PRIVATE rccl_rdi_defs)

# Plain (non-extern) __shared__ in common.h — same rationale as rccl_rdi_host:
# these TUs include common.h but never read ncclShmem; without RDC, newer clang
# rejects `extern __shared__` at file scope.
#
# RCCL_DEVICE_TABLE_OMIT suppresses the __device__ ncclDevFuncTable_*[] array
# (and its references to ncclDevFunc_* symbols that only exist in the per-arch
# specialized pipeline).  onerank.cu.cpp and collectives.cc don't invoke
# ncclKernelMain, so dropping the table is safe and avoids the per-TU
# amdgcn device link trying to resolve those symbols.
target_compile_definitions(rccl_rdi_aux PRIVATE
  RCCL_NO_EXTERN_SHMEM
  RCCL_DEVICE_TABLE_OMIT
)

target_compile_options(rccl_rdi_aux PRIVATE
  -x hip
  ${RDI_OFFLOAD_ARCH_FLAGS}
  ${RDI_OPT_FLAGS}
  -std=c++17
  -fPIC
  -w
  ${RDI_HIP_COMPILER_FLAGS}
  ${RDI_HOST_COMPRESS}
)
add_dependencies(rccl_rdi_aux hipify_all)

# ===========================================================================
# Top-level target: rccl depends on this, and will link the host object in.
# ===========================================================================
add_custom_target(rdc_isa_device_build ALL
  DEPENDS
    $<TARGET_OBJECTS:rccl_rdi_host>
    $<TARGET_OBJECTS:rccl_rdi_aux>
)
add_dependencies(rdc_isa_device_build hipify_all)

set(RDC_ISA_DEVICE_OBJECTS
  $<TARGET_OBJECTS:rccl_rdi_host>
  $<TARGET_OBJECTS:rccl_rdi_aux>
)
