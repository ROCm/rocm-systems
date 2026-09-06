# cmake/DeviceLinker.cmake
#
# Assembly-extract device linker pipeline, using the RCCLDEV custom language.
#
# The rccl-device-compile driver presents a compiler/linker interface to CMake.
# Per-kernel compilation (cpp -> extract -> obj) is a native CMake compile step.
# Per-arch linking (objects -> aggregate -> patch -> link -> elf) uses the driver
# in --link mode via a custom command.
#
# Required variables (set by src/CMakeLists.txt before including this file):
#   HIPIFY_DIR, GEN_DIR, GPU_TARGETS, PROJECT_BINARY_DIR, PROJECT_SOURCE_DIR,
#   Python3_EXECUTABLE

message(STATUS "Device Linker: assembly-extract pipeline enabled (RCCLDEV language)")

# ---------------------------------------------------------------------------
# Enable RCCLDEV custom language
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
enable_language(RCCLDEV)

# Tell the driver where to find the real compiler.
get_filename_component(_dl_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
find_program(DL_CLANG NAMES amdclang++ clang++
  HINTS "${_dl_compiler_dir}" "${ROCM_PATH}/bin" REQUIRED)
find_program(DL_BUNDLER NAMES clang-offload-bundler
  HINTS "${_dl_compiler_dir}" "${_dl_compiler_dir}/../lib/llvm/bin"
        "${ROCM_PATH}/llvm/bin" REQUIRED)

# Extract --hip-path and --hip-device-lib-path from CMAKE_CXX_FLAGS.
# TheRock's amd-hip toolchain injects these so amdclang++ can locate HIP
# headers and device bitcode in its split directory layout.  Standard ROCm
# installs don't set them (amdclang++ auto-discovers from its own location).
# We must forward any that exist to every amdclang++ -x hip invocation we make.
set(DL_HIP_COMPILER_FLAGS "")
string(REGEX MATCHALL "--hip-path=[^ ]+" _hip_path_flags "${CMAKE_CXX_FLAGS}")
list(APPEND DL_HIP_COMPILER_FLAGS ${_hip_path_flags})
string(REGEX MATCHALL "--hip-device-lib-path=[^ ]+" _hip_devlib_flags "${CMAKE_CXX_FLAGS}")
list(APPEND DL_HIP_COMPILER_FLAGS ${_hip_devlib_flags})
if(DL_HIP_COMPILER_FLAGS)
  message(STATUS "Device Linker: forwarding HIP flags from toolchain: ${DL_HIP_COMPILER_FLAGS}")
else()
  message(STATUS "Device Linker: no --hip-path/--hip-device-lib-path in CMAKE_CXX_FLAGS (standard ROCm install)")
endif()

set(DEVICE_BUILD_DIR "${PROJECT_BINARY_DIR}/device_build")
set(SPECIALIZED_DIR  "${GEN_DIR}/specialized")

# ---------------------------------------------------------------------------
# Compile options inherited from the rccl target
#
# This file is included after every target_compile_options(rccl ...) call, so
# the target already carries the flags that govern device codegen -- notably
# -mllvm --amdgpu-kernarg-preload-count=N and -fvisibility=hidden.  The custom
# commands below invoke amdclang++ directly, so without forwarding these they
# silently produce different code than the -fgpu-rdc build.  Losing kernarg
# preloading in particular costs a memory round trip at every kernel entry,
# which is measurable on the small latency-bound kernels (DDA).
#
# Dropped here: flags selecting the compilation model (each command sets its
# own -x hip / --offload-arch, and --offload-host-only would suppress the very
# device code these commands exist to produce), -parallel-jobs (would
# oversubscribe an already parallel build), and diagnostics (generated sources
# are compiled quietly by design, and some need -w).  --offload-compress is
# also filtered out of the rccl target copy below and re-appended when
# ENABLE_COMPRESS is on, so every command using ${DL_INHERITED_FLAGS} picks it
# up consistently.
#
# Also dropped are options that are only meaningful to something other than the
# amdclang++ invocations below: SHELL: is an escaping prefix CMake expands only
# when generating a target's own command line, so forwarding it here would pass
# the literal string through (ENABLE_CODE_COVERAGE adds two), and --hipcc-* are
# hipcc driver options while these commands drive amdclang++ directly.
# ---------------------------------------------------------------------------
set(DL_INHERITED_FLAGS "")
get_target_property(_rccl_copts rccl COMPILE_OPTIONS)
if(_rccl_copts)
  foreach(_opt IN LISTS _rccl_copts)
    if(_opt MATCHES "^(-x|hip|-fgpu-rdc|--offload-host-only|--offload-compress|--offload-arch=.*|-parallel-jobs=.*|-w|-W.*)$")
      continue()
    endif()
    if(_opt MATCHES "^(SHELL:|--hipcc-)")
      continue()
    endif()
    list(APPEND DL_INHERITED_FLAGS "${_opt}")
  endforeach()
endif()
if(ENABLE_COMPRESS AND ROCM_VERSION VERSION_GREATER_EQUAL "60200")
  list(APPEND DL_INHERITED_FLAGS "--offload-compress")
endif()
message(STATUS "Device Linker: inherited compile options: ${DL_INHERITED_FLAGS}")

# ---------------------------------------------------------------------------
# Parse GPU_TARGETS: strip target features, build offload-arch flag list
# ---------------------------------------------------------------------------
set(DL_GPU_TARGETS "")
set(DL_OFFLOAD_ARCH_FLAGS "")
foreach(_gpu_raw ${GPU_TARGETS})
  string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
  list(APPEND DL_GPU_TARGETS "${_gpu}")
  list(APPEND DL_OFFLOAD_ARCH_FLAGS "--offload-arch=${_gpu}")
endforeach()
message(STATUS "Device Linker: GPU targets = ${DL_GPU_TARGETS}")

# ---------------------------------------------------------------------------
# Optimization flags (passed to both compile and link modes of the driver)
# ---------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(DL_OPT_FLAGS -O1 -g)
else()
  set(DL_OPT_FLAGS -O3)
endif()

# ---------------------------------------------------------------------------
# INTERFACE library: shared definitions and includes for device compilation.
# Reads from the rccl target (already fully configured) and directory scope.
# No manual lists — everything comes from what CMake already knows.
# ---------------------------------------------------------------------------
add_library(rccl_device_defs INTERFACE)

# Target-scope definitions from the rccl target
get_target_property(_rccl_defs rccl COMPILE_DEFINITIONS)
if(_rccl_defs)
  target_compile_definitions(rccl_device_defs INTERFACE ${_rccl_defs})
endif()

# Directory-scope definitions (add_compile_definitions / add_definitions in root CMakeLists.txt)
get_directory_property(_dir_defs COMPILE_DEFINITIONS)
if(_dir_defs)
  target_compile_definitions(rccl_device_defs INTERFACE ${_dir_defs})
endif()

# __HIP_PLATFORM_AMD__ and FMT_HEADER_ONLY come from linked targets (hip::device)
# and are not visible via get_target_property. Add them explicitly.
target_compile_definitions(rccl_device_defs INTERFACE
  __HIP_PLATFORM_AMD__=1
  FMT_HEADER_ONLY=1
)

# Include directories from the rccl target (only the device-relevant subset)
get_target_property(_rccl_includes rccl INCLUDE_DIRECTORIES)
if(_rccl_includes)
  target_include_directories(rccl_device_defs INTERFACE ${_rccl_includes})
endif()

# System includes: HIP headers from hip::device (or hip::amdhip64, hip::host).
# We query specific targets rather than iterating all LINK_LIBRARIES because
# some targets use generator expressions in INTERFACE_INCLUDE_DIRECTORIES that
# can't be resolved by get_target_property in manual flag construction.
set(_hip_includes "")
foreach(_hip_tgt hip::device hip::amdhip64 hip::host)
  if(TARGET ${_hip_tgt} AND NOT _hip_includes)
    get_target_property(_hip_includes ${_hip_tgt} INTERFACE_INCLUDE_DIRECTORIES)
  endif()
endforeach()
if(_hip_includes)
  target_include_directories(rccl_device_defs SYSTEM INTERFACE ${_hip_includes})
elseif(ROCM_PATH)
  target_include_directories(rccl_device_defs SYSTEM INTERFACE "${ROCM_PATH}/include")
endif()

# fmt headers: FetchContent provides fmt_SOURCE_DIR; find_package provides the target.
if(fmt_SOURCE_DIR)
  target_include_directories(rccl_device_defs SYSTEM INTERFACE "${fmt_SOURCE_DIR}/include")
elseif(TARGET fmt::fmt-header-only)
  get_target_property(_fmt_inc fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
  if(_fmt_inc)
    foreach(_p ${_fmt_inc})
      if(NOT _p MATCHES "^\\$<")
        target_include_directories(rccl_device_defs SYSTEM INTERFACE "${_p}")
      endif()
    endforeach()
  endif()
endif()

# ---------------------------------------------------------------------------
# Read specialized file list
# ---------------------------------------------------------------------------
set(SPECIALIZED_FILES_TXT "${GEN_DIR}/specialized_files.txt")
if(NOT EXISTS "${SPECIALIZED_FILES_TXT}")
  message(FATAL_ERROR "Device Linker: ${SPECIALIZED_FILES_TXT} not found. generate.py must run first.")
endif()

file(STRINGS "${SPECIALIZED_FILES_TXT}" SPECIALIZED_ENTRIES)
list(LENGTH SPECIALIZED_ENTRIES DL_KERNEL_COUNT)
message(STATUS "Device Linker: ${DL_KERNEL_COUNT} specialized kernels")

# ---------------------------------------------------------------------------
# Guard evaluation: skip kernels whose #if guard excludes a GPU target.
# ---------------------------------------------------------------------------
function(dl_evaluate_guard GUARD GPU_TARGET RESULT_VAR)
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
# Derive host triple for the offload bundler
# ---------------------------------------------------------------------------
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _dl_sys_name)
if(NOT _dl_sys_name)
  set(_dl_sys_name "linux")
endif()
set(_dl_host_triple "${CMAKE_SYSTEM_PROCESSOR}-unknown-${_dl_sys_name}-gnu")

# ===========================================================================
# Per-GPU-target: OBJECT library (compile) + link custom command
# ===========================================================================
set(ALL_DEVICE_ELFS "")
set(DL_BUNDLER_TARGETS "host-${_dl_host_triple}-")
set(DL_BUNDLER_INPUTS "--input=/dev/null")
set(ALL_IR_FILES "")

foreach(DL_GPU_TARGET ${DL_GPU_TARGETS})
  # Sort CDNA targets first for better build scheduling (see original rationale)
  if(DL_GPU_TARGET MATCHES "^gfx9")
    set(DL_ARCH_DIR "${DEVICE_BUILD_DIR}/-${DL_GPU_TARGET}")
  else()
    set(DL_ARCH_DIR "${DEVICE_BUILD_DIR}/${DL_GPU_TARGET}")
  endif()
  file(MAKE_DIRECTORY ${DL_ARCH_DIR})

  # =========================================================================
  # Filter specialized sources for this arch
  # =========================================================================
  set(ARCH_SOURCES "")
  set(_dl_skipped 0)

  foreach(ENTRY ${SPECIALIZED_ENTRIES})
    if(NOT ENTRY MATCHES "^([^ ]+) +([^ ]+) *(.*)")
      continue()
    endif()
    set(CPP_FILE "${CMAKE_MATCH_1}")
    set(_entry_guard "${CMAKE_MATCH_3}")

    dl_evaluate_guard("${_entry_guard}" "${DL_GPU_TARGET}" _guard_ok)
    if(NOT _guard_ok)
      math(EXPR _dl_skipped "${_dl_skipped} + 1")
      continue()
    endif()

    list(APPEND ARCH_SOURCES "${SPECIALIZED_DIR}/${CPP_FILE}")
  endforeach()

  list(LENGTH ARCH_SOURCES _dl_built)
  if(_dl_skipped GREATER 0)
    message(STATUS "Device Linker [${DL_GPU_TARGET}]: ${_dl_built} kernels to build, ${_dl_skipped} skipped (arch guard)")
  endif()

  # =========================================================================
  # OBJECT library: per-kernel device compilation via RCCLDEV language
  # =========================================================================
  set(_dev_target "rccl_device_${DL_GPU_TARGET}")

  add_library(${_dev_target} OBJECT ${ARCH_SOURCES})
  set_source_files_properties(${ARCH_SOURCES} PROPERTIES LANGUAGE RCCLDEV)
  set_target_properties(${_dev_target} PROPERTIES
    LINKER_LANGUAGE RCCLDEV
  )

  target_compile_options(${_dev_target} PRIVATE
    --arch=${DL_GPU_TARGET}
    --clang=${DL_CLANG}
    ${DL_OPT_FLAGS}
    -std=c++17
    ${DL_HIP_COMPILER_FLAGS}
    # -fPIC is required so amdclang++ emits GOT-relative relocations for
    # cross-function calls inside the device .o files. Without it, larger
    # ncclDevFunc_* bodies (e.g. unroll=8/16 reductions on f8e4m3/f8e5m2 or
    # PAT/LL ReduceScatter) exceed the compiler's inlining threshold and
    # produce R_AMDGPU_REL64 references, which `ld.lld -shared` then rejects
    # against default-visibility symbols ("recompile with -fPIC"). Every
    # other device compile step in this file already passes -fPIC; this
    # brings the per-kernel OBJECT build in line with the rest.
    -fPIC
    ${DL_INHERITED_FLAGS}
  )
  target_compile_definitions(${_dev_target} PRIVATE RCCL_DEVICE_LINKER)
  target_link_libraries(${_dev_target} PRIVATE rccl_device_defs)

  add_dependencies(${_dev_target} hipify_all copy_nccl_device_headers)
  if((ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN) AND TARGET rocshmem_static)
    add_dependencies(${_dev_target} rocshmem_static)
  endif()
  # Pass rocshmem device bitcode to per-kernel compiles so rocshmem device
  # symbols resolve during the per-arch device.elf link step.
  # ENABLE_ROCSHMEM: rocshmem_n_pes, alltoall_wg, etc.
  # ENABLE_ROCSHMEM_GIN: QueuePair::put_nbi, atomic_add, etc.
  if((ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN) AND ROCSHMEM_INSTALL_DIR)
    set(_rocshmem_bc "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem_device_${DL_GPU_TARGET}.bc")
    target_compile_options(${_dev_target} PRIVATE --rocshmem-bitcode=${_rocshmem_bc})
  endif()

  # =========================================================================
  # Link step: driver --link mode produces device.elf
  # =========================================================================
  set(ARCH_DEVICE_ELF "${DL_ARCH_DIR}/device.elf")

  # Gather definitions and includes for the dispatcher compilation inside --link.
  # The driver forwards these to amdclang++ when compiling common.cu.cpp.
  get_target_property(_dev_defs ${_dev_target} COMPILE_DEFINITIONS)
  set(_link_def_flags "")
  if(_dev_defs)
    foreach(_d ${_dev_defs})
      list(APPEND _link_def_flags "-D${_d}")
    endforeach()
  endif()
  # Also add interface definitions from rccl_device_defs
  get_target_property(_iface_defs rccl_device_defs INTERFACE_COMPILE_DEFINITIONS)
  if(_iface_defs)
    foreach(_d ${_iface_defs})
      list(APPEND _link_def_flags "-D${_d}")
    endforeach()
  endif()
  list(REMOVE_DUPLICATES _link_def_flags)

  get_target_property(_dev_includes rccl_device_defs INTERFACE_INCLUDE_DIRECTORIES)
  set(_link_inc_flags "")
  if(_dev_includes)
    foreach(_inc ${_dev_includes})
      list(APPEND _link_inc_flags "-I${_inc}")
    endforeach()
  endif()
  get_target_property(_dev_sys_includes rccl_device_defs INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
  if(_dev_sys_includes)
    foreach(_inc ${_dev_sys_includes})
      if(NOT _inc IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
        list(APPEND _link_inc_flags "-isystem${_inc}")
      endif()
    endforeach()
  endif()

  set(_link_rsp "${DL_ARCH_DIR}/link_objects.rsp")
  file(GENERATE OUTPUT "${_link_rsp}"
    CONTENT "$<JOIN:$<TARGET_OBJECTS:${_dev_target}>,\n>\n")

  # When rocSHMEM is enabled, pass the per-arch device bitcode to the driver.
  # rocSHMEM device API symbols have hidden visibility and must be statically
  # present in the device ELF — they cannot be imported from a shared library.
  set(_rocshmem_bitcode_arg "")
  set(_rocshmem_link_depends "")
  # Not ENABLE_ROCSHMEM_GIN: no object here references rocSHMEM, and the full
  # device bitcode carries unresolved IPC and RO symbols that break the ELF load.
  if(ENABLE_ROCSHMEM AND ROCSHMEM_INSTALL_DIR)
    set(_rocshmem_bc "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem_device_${DL_GPU_TARGET}.bc")
    set(_rocshmem_bitcode_arg "--rocshmem-bitcode=${_rocshmem_bc}")
    if(TARGET rocshmem_static)
      list(APPEND _rocshmem_link_depends rocshmem_static)
    endif()
  endif()

  add_custom_command(
    OUTPUT  ${ARCH_DEVICE_ELF}
    COMMAND ${CMAKE_RCCLDEV_COMPILER}
      --link
      --arch=${DL_GPU_TARGET}
      --clang=${DL_CLANG}
      ${DL_HIP_COMPILER_FLAGS}
      --dispatcher=${HIPIFY_DIR}/src/device/common.cu.cpp
      ${_rocshmem_bitcode_arg}
      ${_link_def_flags}
      ${_link_inc_flags}
      ${DL_OPT_FLAGS}
      -std=c++17
      -o ${ARCH_DEVICE_ELF}
      @${_link_rsp}
    DEPENDS ${_dev_target} ${HIPIFY_DIR}/src/device/common.cu.cpp ${_rocshmem_link_depends}
    COMMENT "DL [${DL_GPU_TARGET}] link: device.elf"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )

  list(APPEND ALL_DEVICE_ELFS "${ARCH_DEVICE_ELF}")
  list(APPEND DL_BUNDLER_TARGETS "hip-amdgcn-amd-amdhsa--${DL_GPU_TARGET}")
  list(APPEND DL_BUNDLER_INPUTS "--input=${ARCH_DEVICE_ELF}")

  # =========================================================================
  # Optional: emit LLVM IR for specialized kernels (ninja device_ir)
  # =========================================================================
  set(DL_ARCH_IR_DIR "${DL_ARCH_DIR}/device_ir")
  file(MAKE_DIRECTORY ${DL_ARCH_IR_DIR})

  foreach(ENTRY ${SPECIALIZED_ENTRIES})
    if(NOT ENTRY MATCHES "^([^ ]+) +([^ ]+) *(.*)")
      continue()
    endif()
    set(CPP_FILE "${CMAKE_MATCH_1}")
    set(_entry_guard "${CMAKE_MATCH_3}")
    string(REGEX REPLACE "\\.cpp$" "" BASE "${CPP_FILE}")

    dl_evaluate_guard("${_entry_guard}" "${DL_GPU_TARGET}" _guard_ok)
    if(NOT _guard_ok)
      continue()
    endif()

    set(SRC     "${SPECIALIZED_DIR}/${CPP_FILE}")
    set(IR_OUT  "${DL_ARCH_IR_DIR}/${BASE}.ll")

    add_custom_command(
      OUTPUT  ${IR_OUT}
      COMMAND ${DL_CLANG}
        -DRCCL_DEVICE_LINKER
        ${_link_def_flags}
        ${_link_inc_flags}
        -x hip --offload-device-only --offload-arch=${DL_GPU_TARGET}
        ${DL_HIP_COMPILER_FLAGS}
        -gline-tables-only
        -std=c++17 ${DL_OPT_FLAGS}
        -emit-llvm -S
        -o ${IR_OUT}
        ${SRC}
      DEPENDS ${SRC}
      COMMENT "DL [${DL_GPU_TARGET}] IR: ${CPP_FILE}"
      VERBATIM
    )
    list(APPEND ALL_IR_FILES ${IR_OUT})
  endforeach()

endforeach()  # end per-GPU-target loop

# ===========================================================================
# Bundle all per-arch device.elf files into a single .hipfb fat binary
# ===========================================================================
set(DEVICE_HIPFB "${DEVICE_BUILD_DIR}/device.hipfb")

list(JOIN DL_BUNDLER_TARGETS "," _bundler_targets_str)

set(DL_BUNDLER_COMPRESS "")
if(ENABLE_COMPRESS)
  set(DL_BUNDLER_COMPRESS "--compress")
endif()

add_custom_command(
  OUTPUT  ${DEVICE_HIPFB}
  COMMAND ${DL_BUNDLER}
    --type=bc
    --targets=${_bundler_targets_str}
    ${DL_BUNDLER_INPUTS}
    --output=${DEVICE_HIPFB}
    ${DL_BUNDLER_COMPRESS}
  DEPENDS ${ALL_DEVICE_ELFS}
  COMMENT "DL bundle: device.elf(s) -> device.hipfb [${DL_GPU_TARGETS}]"
  VERBATIM
)

# ===========================================================================
# Host compile common.cu.cpp with embedded device binary
# ===========================================================================
set(COMMON_FAT_OBJ "${DEVICE_BUILD_DIR}/common.o")

# Gather include flags for host compile (same paths as device)
set(_host_inc_flags "")
if(_dev_includes)
  foreach(_inc ${_dev_includes})
    list(APPEND _host_inc_flags "-I${_inc}")
  endforeach()
endif()
if(_dev_sys_includes)
  foreach(_inc ${_dev_sys_includes})
    if(NOT _inc IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
      list(APPEND _host_inc_flags "-isystem${_inc}")
    endif()
  endforeach()
endif()

add_custom_command(
  OUTPUT  ${COMMON_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip --offload-host-only ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -Xclang -fcuda-include-gpubinary -Xclang ${DEVICE_HIPFB}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -c -o ${COMMON_FAT_OBJ}
    ${HIPIFY_DIR}/src/device/common.cu.cpp
  DEPENDS ${DEVICE_HIPFB} ${HIPIFY_DIR}/src/device/common.cu.cpp
  COMMENT "DL host compile: common.cu.cpp with embedded device binary"
  VERBATIM
)

# ===========================================================================
# Onerank: normal HIP compilation (host+device, no RDC)
# ===========================================================================
set(ONERANK_FAT_OBJ "${DEVICE_BUILD_DIR}/onerank.o")

add_custom_command(
  OUTPUT  ${ONERANK_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -c -o ${ONERANK_FAT_OBJ}
    ${HIPIFY_DIR}/src/device/onerank.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/device/onerank.cu.cpp
  COMMENT "DL compile: onerank.cu.cpp (normal fat object)"
  VERBATIM
)

# ===========================================================================
# collectives.cc: contains a __global__ kernel launch (hierarchicalShuffle)
# so it needs full HIP compilation, not --offload-host-only.
# ===========================================================================
# Dependency tracking note (CMake >= 3.20 vs < 3.20):
# add_custom_command only rebuilds when files listed in DEPENDS change.  It
# does NOT automatically track transitive headers (e.g. ce_coll.h), so a
# struct layout change in a header would not invalidate collectives.o.
# CMake 3.20 DEPFILE support fixes this by reading the compiler-generated .d
# file; on older CMake the workaround is: touch hipify/src/collectives.cc.
# ===========================================================================
set(COLLECTIVES_FAT_OBJ "${DEVICE_BUILD_DIR}/collectives.o")

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
  set(COLLECTIVES_DEPFILE "${DEVICE_BUILD_DIR}/collectives.d")
  add_custom_command(
    OUTPUT  ${COLLECTIVES_FAT_OBJ}
    COMMAND ${DL_CLANG}
      -x hip ${DL_OFFLOAD_ARCH_FLAGS}
      ${DL_HIP_COMPILER_FLAGS}
      -DRCCL_DEVICE_LINKER
      ${_link_def_flags}
      ${_host_inc_flags}
      ${DL_OPT_FLAGS}
      ${DL_INHERITED_FLAGS}
      -std=c++17
      -fPIC
      -MD -MF ${COLLECTIVES_DEPFILE}
      -c -o ${COLLECTIVES_FAT_OBJ}
      ${HIPIFY_DIR}/src/collectives.cc
    DEPENDS ${HIPIFY_DIR}/src/collectives.cc
    DEPFILE ${COLLECTIVES_DEPFILE}
    COMMENT "DL compile: collectives.cc (has __global__ kernel, header-tracking via DEPFILE)"
    VERBATIM
  )
else()
  add_custom_command(
    OUTPUT  ${COLLECTIVES_FAT_OBJ}
    COMMAND ${DL_CLANG}
      -x hip ${DL_OFFLOAD_ARCH_FLAGS}
      ${DL_HIP_COMPILER_FLAGS}
      -DRCCL_DEVICE_LINKER
      ${_link_def_flags}
      ${_host_inc_flags}
      ${DL_OPT_FLAGS}
      ${DL_INHERITED_FLAGS}
      -std=c++17
      -fPIC
      -c -o ${COLLECTIVES_FAT_OBJ}
      ${HIPIFY_DIR}/src/collectives.cc
    DEPENDS ${HIPIFY_DIR}/src/collectives.cc
    COMMENT "DL compile: collectives.cc (has __global__ kernel)"
    VERBATIM
  )
endif()

# ===========================================================================
# dda_all_reduce_ipc.cu.cpp: contains device kernels and kernel launches.
# Like collectives.cc, it cannot be built with --offload-host-only on the
# main rccl target or __hip_fatbin_* stays undefined in librccl.so.
# ===========================================================================
set(DDA_ALL_REDUCE_IPC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_reduce_ipc.o")
set(DDA_REDUCE_SCATTER_IPC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_reduce_scatter_ipc.o")
set(DDA_ALL_GATHER_IPC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_gather_ipc.o")
set(DDA_ALLTOALL_IPC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_alltoall_ipc.o")

add_custom_command(
  OUTPUT  ${DDA_ALL_REDUCE_IPC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -c -o ${DDA_ALL_REDUCE_IPC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_ipc.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_ipc.cu.cpp
  COMMENT "DL compile: dda_all_reduce_ipc.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_REDUCE_SCATTER_IPC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -c -o ${DDA_REDUCE_SCATTER_IPC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_ipc.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_ipc.cu.cpp
  COMMENT "DL compile: dda_reduce_scatter_ipc.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALL_GATHER_IPC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -c -o ${DDA_ALL_GATHER_IPC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_ipc.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_ipc.cu.cpp
  COMMENT "DL compile: dda_all_gather_ipc.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALLTOALL_IPC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -c -o ${DDA_ALLTOALL_IPC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_ipc.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_ipc.cu.cpp
  COMMENT "DL compile: dda_alltoall_ipc.cu.cpp (has device kernels)"
  VERBATIM
)

# ===========================================================================
# gin_alltoall_sdma.cu.cpp: GIN-SDMA alltoall kernel.
#
# The defines go here because this file is filtered out of the rccl target, so
# per-source properties never apply. Leaving only SDMA on keeps ncclGinCallImpl
# on its single backend branch. GDA must stay off, as this object gets no QP bitcode.
# ===========================================================================
set(GIN_ALLTOALL_SDMA_FAT_OBJ "")
if(ENABLE_ROCSHMEM_GIN)
  set(GIN_ALLTOALL_SDMA_FAT_OBJ "${DEVICE_BUILD_DIR}/gin_alltoall_sdma.o")
  add_custom_command(
    OUTPUT  ${GIN_ALLTOALL_SDMA_FAT_OBJ}
    COMMAND ${DL_CLANG}
      -x hip ${DL_OFFLOAD_ARCH_FLAGS}
      ${DL_HIP_COMPILER_FLAGS}
      -DRCCL_DEVICE_LINKER
      -DNCCL_GIN_ANVIL_SDMA_ENABLE=1
      -DNCCL_GIN_PROXY_ENABLE=0
      -DNCCL_GIN_ROCSHMEM_GDA_ENABLE=0
      ${_link_def_flags}
      ${_host_inc_flags}
      ${DL_OPT_FLAGS}
      ${DL_INHERITED_FLAGS}
      -std=c++17
      -fPIC
      -c -o ${GIN_ALLTOALL_SDMA_FAT_OBJ}
      ${HIPIFY_DIR}/src/algorithms/gin/sdma/gin_alltoall_sdma.cu.cpp
    DEPENDS ${HIPIFY_DIR}/src/algorithms/gin/sdma/gin_alltoall_sdma.cu.cpp
    COMMENT "DL compile: gin_alltoall_sdma.cu.cpp (GIN-SDMA alltoall kernel)"
    VERBATIM
  )
endif()

# ===========================================================================
# gin_all_reduce_sdma.cu.cpp: GIN-SDMA allreduce kernel.
#
# The defines go here because this file is filtered out of the rccl target, so
# per-source properties never apply. Leaving only SDMA on keeps ncclGinCallImpl
# on its single backend branch. GDA must stay off, as this object gets no QP bitcode.
# ===========================================================================
set(GIN_ALLREDUCE_SDMA_FAT_OBJ "")
if(ENABLE_ROCSHMEM_GIN)
  set(GIN_ALLREDUCE_SDMA_FAT_OBJ "${DEVICE_BUILD_DIR}/gin_all_reduce_sdma.o")
  add_custom_command(
    OUTPUT  ${GIN_ALLREDUCE_SDMA_FAT_OBJ}
    COMMAND ${DL_CLANG}
      -x hip ${DL_OFFLOAD_ARCH_FLAGS}
      ${DL_HIP_COMPILER_FLAGS}
      -DRCCL_DEVICE_LINKER
      -DNCCL_GIN_ANVIL_SDMA_ENABLE=1
      -DNCCL_GIN_PROXY_ENABLE=0
      -DNCCL_GIN_ROCSHMEM_GDA_ENABLE=0
      ${_link_def_flags}
      ${_host_inc_flags}
      ${DL_OPT_FLAGS}
      ${DL_INHERITED_FLAGS}
      -std=c++17
      -fPIC
      -c -o ${GIN_ALLREDUCE_SDMA_FAT_OBJ}
      ${HIPIFY_DIR}/src/algorithms/gin/sdma/gin_all_reduce_sdma.cu.cpp
    DEPENDS ${HIPIFY_DIR}/src/algorithms/gin/sdma/gin_all_reduce_sdma.cu.cpp
    COMMENT "DL compile: gin_all_reduce_sdma.cu.cpp (GIN-SDMA allreduce kernel)"
    VERBATIM
  )
endif()
# ===========================================================================
# dda_all_reduce_fabric.cu.cpp: fabric/VMM counterpart of the IPC file above.
# ===========================================================================
set(DDA_ALL_REDUCE_FABRIC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_reduce_fabric.o")

add_custom_command(
  OUTPUT  ${DDA_ALL_REDUCE_FABRIC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${DDA_ALL_REDUCE_FABRIC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric.cu.cpp
  COMMENT "DL compile: dda_all_reduce_fabric.cu.cpp (has device kernels)"
  VERBATIM
)

set(DDA_ALL_REDUCE_FABRIC_LL_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_reduce_fabric_ll.o")
set(DDA_ALL_REDUCE_FABRIC_LL128_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_reduce_fabric_ll128.o")

add_custom_command(
  OUTPUT  ${DDA_ALL_REDUCE_FABRIC_LL_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_ALL_REDUCE_FABRIC_LL_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll.cu.cpp
  COMMENT "DL compile: dda_all_reduce_fabric_ll.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALL_REDUCE_FABRIC_LL128_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_ALL_REDUCE_FABRIC_LL128_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll128.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll128.cu.cpp
  COMMENT "DL compile: dda_all_reduce_fabric_ll128.cu.cpp (has device kernels)"
  VERBATIM
)

set(DDA_REDUCE_SCATTER_FABRIC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_reduce_scatter_fabric.o")
set(DDA_ALL_GATHER_FABRIC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_gather_fabric.o")
set(DDA_ALL_GATHER_FABRIC_LL_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_gather_fabric_ll.o")
set(DDA_ALL_GATHER_FABRIC_LL128_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_all_gather_fabric_ll128.o")
set(DDA_ALLTOALL_FABRIC_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_alltoall_fabric.o")
set(DDA_ALLTOALL_FABRIC_LL_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_alltoall_fabric_ll.o")
set(DDA_ALLTOALL_FABRIC_LL128_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_alltoall_fabric_ll128.o")
set(DDA_REDUCE_SCATTER_FABRIC_LL_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_reduce_scatter_fabric_ll.o")
set(DDA_REDUCE_SCATTER_FABRIC_LL128_FAT_OBJ "${DEVICE_BUILD_DIR}/dda_reduce_scatter_fabric_ll128.o")

add_custom_command(
  OUTPUT  ${DDA_REDUCE_SCATTER_FABRIC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${DDA_REDUCE_SCATTER_FABRIC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric.cu.cpp
  COMMENT "DL compile: dda_reduce_scatter_fabric.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALL_GATHER_FABRIC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${DDA_ALL_GATHER_FABRIC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric.cu.cpp
  COMMENT "DL compile: dda_all_gather_fabric.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALL_GATHER_FABRIC_LL_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_ALL_GATHER_FABRIC_LL_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric_ll.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric_ll.cu.cpp
  COMMENT "DL compile: dda_all_gather_fabric_ll.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALL_GATHER_FABRIC_LL128_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_ALL_GATHER_FABRIC_LL128_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric_ll128.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/all_gather/dda_all_gather_fabric_ll128.cu.cpp
  COMMENT "DL compile: dda_all_gather_fabric_ll128.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALLTOALL_FABRIC_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${DDA_ALLTOALL_FABRIC_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric.cu.cpp
  COMMENT "DL compile: dda_alltoall_fabric.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALLTOALL_FABRIC_LL_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_ALLTOALL_FABRIC_LL_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu.cpp
  COMMENT "DL compile: dda_alltoall_fabric_ll.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_ALLTOALL_FABRIC_LL128_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_ALLTOALL_FABRIC_LL128_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll128.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/alltoall/dda_alltoall_fabric_ll128.cu.cpp
  COMMENT "DL compile: dda_alltoall_fabric_ll128.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_REDUCE_SCATTER_FABRIC_LL_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_REDUCE_SCATTER_FABRIC_LL_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll.cu.cpp
  COMMENT "DL compile: dda_reduce_scatter_fabric_ll.cu.cpp (has device kernels)"
  VERBATIM
)

add_custom_command(
  OUTPUT  ${DDA_REDUCE_SCATTER_FABRIC_LL128_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    -std=c++17
    -fPIC
    ${DL_INHERITED_FLAGS}
    -w
    -c -o ${DDA_REDUCE_SCATTER_FABRIC_LL128_FAT_OBJ}
    ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu.cpp
  COMMENT "DL compile: dda_reduce_scatter_fabric_ll128.cu.cpp (has device kernels)"
  VERBATIM
)

# ===========================================================================
# CE-reduce kernels: per-instantiation device TUs from gensrc/ce_reduce/.
# Each instantiation file defines one ncclCeLocalReduceKernelVec<T,RedOp,U>
# (__global__) and its host-callable launcher. Compiled with full HIP here so
# each fat binary is self-contained (ce_coll.cc, the main target, has no
# __global__ call sites and stays --offload-host-only).
#
# CE_REDUCE_FAT_OBJS is plural (mirroring SYM_FAT_OBJS below) because
# src/device/ce_reduce/generate.py emits one TU per (type, redop)
# instantiation instead of one aggregate ce_reduce.cc -- see that script for
# why: two of the 40 instantiations (int8_t/uint8_t Min/Max) individually
# generate ~56K instructions each and used to dominate the whole build's
# wall-clock time by serializing all 40 kernels' codegen into one TU.
# ===========================================================================
set(CE_REDUCE_FAT_OBJS "")
file(GLOB _ce_reduce_srcs CONFIGURE_DEPENDS "${HIPIFY_DIR}/gensrc/ce_reduce/*.cpp")
foreach(_ce_reduce_src IN LISTS _ce_reduce_srcs)
  get_filename_component(_ce_reduce_name "${_ce_reduce_src}" NAME_WE)
  set(_ce_reduce_obj "${DEVICE_BUILD_DIR}/${_ce_reduce_name}.o")
  add_custom_command(
    OUTPUT  ${_ce_reduce_obj}
    COMMAND ${DL_CLANG}
      -x hip ${DL_OFFLOAD_ARCH_FLAGS}
      ${DL_HIP_COMPILER_FLAGS}
      -DRCCL_DEVICE_LINKER
      ${_link_def_flags}
      ${_host_inc_flags}
      ${DL_OPT_FLAGS}
      ${DL_INHERITED_FLAGS}
      -std=c++17
      -fPIC
      ${DL_INHERITED_FLAGS}
      -w
      -c -o ${_ce_reduce_obj}
      ${_ce_reduce_src}
    DEPENDS ${_ce_reduce_src}
    COMMENT "DL compile: ${_ce_reduce_name} (CE AllReduce reduce kernel)"
    VERBATIM
  )
  list(APPEND CE_REDUCE_FAT_OBJS ${_ce_reduce_obj})
endforeach()

# ===========================================================================
# Symmetric kernels: per-instantiation device TUs from gensrc/symmetric/.
# Each instantiation file defines a handful of __global__ ncclSymkDevKernel_*
# entries. Compiled standalone as multi-arch fat objects, mirroring onerank.o.
#
# SYM_FAT_OBJS is plural (vs the singular COMMON/ONERANK/COLLECTIVES_FAT_OBJ
# siblings) because the symmetric generator emits one TU per instantiation.
# ===========================================================================
set(SYM_FAT_OBJS "")
if(GENERATE_SYM_KERNELS)
  # When ENABLE_ROCSHMEM_GIN is set, GIN device templates reference rocshmem
  # device symbols (QueuePair::put_nbi, atomic_add, etc.). The installed per-arch
  # .bc files are arch-optimized (opt -mcpu=) and can't be used with
  # -mlink-builtin-bitcode across archs. Instead, llvm-link the pre-opt
  # individual source .bc files (arch-agnostic) into a minimal QP-only bitcode.
  set(_sym_rocshmem_bc_flag "")
  set(_sym_rocshmem_deps "")
  if(ENABLE_ROCSHMEM_GIN AND ROCSHMEM_SOURCE_DIR)
    # Pick the first arch's pre-opt bitcode dir (all archs produce identical
    # unoptimized IR since -Xclang -disable-llvm-passes is used).
    list(GET DL_GPU_TARGETS 0 _bc_arch)
    set(_bc_dir "${ROCSHMEM_SOURCE_DIR}/build/bitcode/${_bc_arch}")
    set(_qp_bc "${DEVICE_BUILD_DIR}/rocshmem_qp_device.bc")
    find_program(_llvm_link llvm-link HINTS ${ROCM_PATH}/llvm/bin REQUIRED)
    find_program(_llvm_dis  llvm-dis  HINTS ${ROCM_PATH}/llvm/bin REQUIRED)
    find_program(_llvm_as   llvm-as   HINTS ${ROCM_PATH}/llvm/bin REQUIRED)

    # Pipeline:
    #  1. Compile gin_rocshmem_constmem.hip → device-only .bc (provides
    #     rocshmem::constmem and rocshmem::logd_constants definitions that
    #     queue_pair.bc references as external)
    #  2. llvm-link QP .bc files + constmem .bc into one module
    #  3. Strip @llvm.compiler.used and @__hip_cuid_ via text round-trip
    #     (these AMDGCN addrspace(1) appending globals clash with the
    #     host-side addrspace(0) equivalents in fat-object compilation)
    set(_cm_src "${CMAKE_SOURCE_DIR}/src/gin/gin_rocshmem_constmem.hip")
    set(_cm_bc  "${DEVICE_BUILD_DIR}/gin_rocshmem_constmem.bc")
    set(_qp_raw "${DEVICE_BUILD_DIR}/rocshmem_qp_raw.bc")

    add_custom_command(
      OUTPUT ${_cm_bc}
      COMMAND ${DL_CLANG}
        -x hip --cuda-device-only --offload-arch=${_bc_arch}
        -emit-llvm -Xclang -disable-llvm-passes
        -std=c++17 -fPIC
        -I${ROCSHMEM_SOURCE_DIR}/src
        -I${ROCSHMEM_SOURCE_DIR}/include
        -c -o ${_cm_bc} ${_cm_src}
      DEPENDS ${_cm_src}
      COMMENT "DL: compiling rocshmem constmem stubs to device bitcode"
      VERBATIM)

    add_custom_command(
      OUTPUT ${_qp_bc}
      COMMAND ${_llvm_link}
        ${_bc_dir}/queue_pair.bc
        ${_bc_dir}/queue_pair_mlx5.bc
        ${_bc_dir}/queue_pair_bnxt.bc
        ${_bc_dir}/queue_pair_ionic.bc
        ${_cm_bc}
        -o ${_qp_raw}
      COMMAND ${_llvm_dis} -o ${_qp_raw}.ll ${_qp_raw}
      COMMAND grep -v -E "@llvm[.]compiler[.]used|@__hip_cuid_"
        ${_qp_raw}.ll > ${_qp_raw}.clean.ll
      COMMAND ${_llvm_as} ${_qp_raw}.clean.ll -o ${_qp_bc}
      DEPENDS ${_cm_bc}
      COMMENT "DL: linking rocshmem QP device bitcode (with constmem, stripped)"
      VERBATIM)
    add_custom_target(dl_rocshmem_qp_bc DEPENDS ${_qp_bc})
    if(TARGET rocshmem_static)
      add_dependencies(dl_rocshmem_qp_bc rocshmem_static)
    endif()
    set(_sym_rocshmem_bc_flag -Xclang -mlink-builtin-bitcode -Xclang ${_qp_bc})
    set(_sym_rocshmem_deps dl_rocshmem_qp_bc)
  endif()
  file(GLOB _sym_srcs CONFIGURE_DEPENDS "${HIPIFY_DIR}/gensrc/symmetric/*.cpp")
  foreach(_sym_src IN LISTS _sym_srcs)
    get_filename_component(_sym_name "${_sym_src}" NAME_WE)
    set(_sym_obj "${DEVICE_BUILD_DIR}/sym_${_sym_name}.o")
    # Only GIN symmetric kernels need the QP bitcode; non-GIN ones would
    # just ingest and DCE it, wasting compile time.
    set(_this_bc_flag "")
    set(_this_bc_deps "")
    if(_sym_name MATCHES "_gin_")
      set(_this_bc_flag "${_sym_rocshmem_bc_flag}")
      set(_this_bc_deps "${_sym_rocshmem_deps}")
    endif()
    add_custom_command(
      OUTPUT  ${_sym_obj}
      COMMAND ${DL_CLANG}
        -x hip ${DL_OFFLOAD_ARCH_FLAGS}
        ${DL_HIP_COMPILER_FLAGS}
        -DRCCL_DEVICE_LINKER
        ${_link_def_flags}
        ${_host_inc_flags}
        ${DL_OPT_FLAGS}
        ${DL_INHERITED_FLAGS}
        -std=c++17
        -fPIC
        ${_this_bc_flag}
        -c -o ${_sym_obj}
        ${_sym_src}
      DEPENDS ${_sym_src} ${_this_bc_deps}
      COMMENT "DL compile: sym ${_sym_name} (multi-arch fat object)"
      VERBATIM
    )
    list(APPEND SYM_FAT_OBJS ${_sym_obj})
  endforeach()
endif()

# ===========================================================================
# Top-level target
# ===========================================================================
add_custom_target(device_linker_build ALL
	DEPENDS ${COMMON_FAT_OBJ} ${ONERANK_FAT_OBJ} ${COLLECTIVES_FAT_OBJ} ${DDA_ALL_REDUCE_IPC_FAT_OBJ} ${DDA_REDUCE_SCATTER_IPC_FAT_OBJ} ${DDA_ALL_GATHER_IPC_FAT_OBJ} ${DDA_ALLTOALL_IPC_FAT_OBJ} ${DDA_ALL_REDUCE_FABRIC_FAT_OBJ} ${DDA_ALL_REDUCE_FABRIC_LL_FAT_OBJ} ${DDA_ALL_REDUCE_FABRIC_LL128_FAT_OBJ} ${DDA_REDUCE_SCATTER_FABRIC_FAT_OBJ} ${DDA_ALL_GATHER_FABRIC_FAT_OBJ} ${DDA_ALL_GATHER_FABRIC_LL_FAT_OBJ} ${DDA_ALL_GATHER_FABRIC_LL128_FAT_OBJ} ${DDA_ALLTOALL_FABRIC_FAT_OBJ} ${DDA_ALLTOALL_FABRIC_LL_FAT_OBJ} ${DDA_ALLTOALL_FABRIC_LL128_FAT_OBJ} ${DDA_REDUCE_SCATTER_FABRIC_LL_FAT_OBJ} ${DDA_REDUCE_SCATTER_FABRIC_LL128_FAT_OBJ} ${CE_REDUCE_FAT_OBJS} ${SYM_FAT_OBJS} ${GIN_ALLTOALL_SDMA_FAT_OBJ} ${GIN_ALLREDUCE_SDMA_FAT_OBJ}
)
add_dependencies(device_linker_build hipify_all copy_nccl_device_headers)
if((ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN) AND TARGET rocshmem_static)
  # The fat objects above include GIN device headers, which pull in rocSHMEM
  # headers installed to ext/rocshmem/include by the ExternalProject.
  add_dependencies(device_linker_build rocshmem_static)
endif()

set(DEVICE_LINKER_OBJECTS
  ${COMMON_FAT_OBJ}
  ${ONERANK_FAT_OBJ}
  ${COLLECTIVES_FAT_OBJ}
  ${CE_REDUCE_FAT_OBJS}
  ${DDA_ALL_REDUCE_IPC_FAT_OBJ}
  ${DDA_REDUCE_SCATTER_IPC_FAT_OBJ}
  ${DDA_ALL_GATHER_IPC_FAT_OBJ}
  ${DDA_ALLTOALL_IPC_FAT_OBJ}
  ${DDA_ALL_REDUCE_FABRIC_FAT_OBJ}
  ${DDA_ALL_REDUCE_FABRIC_LL_FAT_OBJ}
  ${DDA_ALL_REDUCE_FABRIC_LL128_FAT_OBJ}
  ${DDA_REDUCE_SCATTER_FABRIC_FAT_OBJ}
  ${DDA_ALL_GATHER_FABRIC_FAT_OBJ}
  ${DDA_ALL_GATHER_FABRIC_LL_FAT_OBJ}
  ${DDA_ALL_GATHER_FABRIC_LL128_FAT_OBJ}
  ${DDA_ALLTOALL_FABRIC_FAT_OBJ}
  ${DDA_ALLTOALL_FABRIC_LL_FAT_OBJ}
  ${DDA_ALLTOALL_FABRIC_LL128_FAT_OBJ}
  ${DDA_REDUCE_SCATTER_FABRIC_LL_FAT_OBJ}
  ${DDA_REDUCE_SCATTER_FABRIC_LL128_FAT_OBJ}
  ${SYM_FAT_OBJS}
  ${GIN_ALLTOALL_SDMA_FAT_OBJ}
  ${GIN_ALLREDUCE_SDMA_FAT_OBJ}
)

# ===========================================================================
# Optional: emit LLVM IR (ninja device_ir)
# ===========================================================================
add_custom_target(device_ir DEPENDS ${ALL_IR_FILES})
add_dependencies(device_ir hipify_all copy_nccl_device_headers)
