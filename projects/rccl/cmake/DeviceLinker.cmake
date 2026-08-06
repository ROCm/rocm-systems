# cmake/DeviceLinker.cmake
#
# Assembly-extract device linker pipeline, using the RCCLDEV custom language.
#
# The rccl-device-compile driver presents a compiler/linker interface to CMake.
# Per-kernel compilation (cpp -> extract -> obj) is a native CMake compile step.
# Per-arch linking (objects -> aggregate -> patch -> link -> elf) uses the driver
# in --link mode via a custom command.
#
# Every device image (the specialized dispatcher kernels below, AND the
# kernels living in mixed host+device TUs like onerank/collectives/dda_*/
# sym_*/ce_reduce) is unified into ONE device.elf per arch, bundled into ONE
# device.hipfb, and embedded exactly ONCE in the final librccl.so by a small
# "glue" object. See the "Shared CUID" and "Mixed host+device TU" sections
# below for how that dedup works.
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
find_program(DL_READELF NAMES llvm-readelf
  HINTS "${_dl_compiler_dir}" "${_dl_compiler_dir}/../lib/llvm/bin"
        "${ROCM_PATH}/llvm/bin" REQUIRED)
find_program(DL_OBJCOPY NAMES llvm-objcopy
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
file(MAKE_DIRECTORY "${DEVICE_BUILD_DIR}")

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
# oversubscribe an already parallel build), --offload-compress (packaging, see
# ENABLE_COMPRESS below) and diagnostics (generated sources are compiled
# quietly by design, and some need -w).
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
# Definitions and include flags shared by every out-of-line amdclang++
# invocation below (mixed-TU host/device compiles, dispatcher --link,
# IR emission). Computed once here since none of it varies per arch.
# ---------------------------------------------------------------------------
set(_link_def_flags "")
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
# Mixed-TU host/device compiles use the same include set as the dispatcher link.
set(_host_inc_flags ${_link_inc_flags})

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
# Shared CUID: fatbin deduplication for mixed host+device TUs
#
# Fully embedding every mixed TU's own device kernels (the historical
# approach: plain `-x hip` full compilation) makes each TU generate and
# embed its OWN self-contained fatbin -- one copy of the runtime's fatbin
# format PER TU. `.hip_fatbin` is a plain PROGBITS section (no COMDAT), so
# the final linker just concatenates all these copies with padding instead
# of deduping them: a build with N mixed TUs pays for N copies of (that TU's
# kernels), which is the actual size blowup this file exists to fix.
#
# Fix: force every mixed TU's HOST compile to use the same fixed `-cuid`
# string (`DL_SHARED_CUID`). HIP's frontend derives the `__hip_fatbin_<hash>`
# symbol name (and a `__hip_cuid_<hash>` "duplicate CUID misuse" marker
# symbol) purely from the `-cuid` string -- independent of the source
# filename or --offload-arch list (verified via probe below) -- so every
# mixed TU ends up referencing the exact same external symbol instead of
# defining its own. Combined with `--offload-host-only` (so the host compile
# emits no fatbin of its own at all, just an undefined reference) and one
# "glue" object that defines that symbol exactly once with the real combined
# fatbin bytes (see the glue.o custom command after DEVICE_HIPFB below), the
# unified image ends up embedded exactly once in librccl.so.
#
# `--offload-host-only` compiles never define `__hip_cuid_<hash>` themselves
# only to complain -- HIP still emits that marker as a GLOBAL symbol on every
# TU that references the fatbin, so linking N mixed TUs together still hits
# a duplicate-symbol error on `__hip_cuid_<hash>` even though they're
# deliberately sharing the same image. `llvm-objcopy --localize-symbol` on
# every mixed TU's raw host object (see the per-TU loop after glue.o) strips
# that marker down to local, since we don't need the misuse detector once
# the sharing is intentional.
# ===========================================================================
set(DL_SHARED_CUID "RCCL_SHARED_DEVICE_IMAGE")

set(_cuid_probe_src "${DEVICE_BUILD_DIR}/cuid_probe.hip")
set(_cuid_probe_obj "${DEVICE_BUILD_DIR}/cuid_probe.o")
file(WRITE "${_cuid_probe_src}"
"#include <hip/hip_runtime.h>
__global__ void dl_cuid_probe_kernel(int* p) { *p = 1; }
extern \"C\" void dl_cuid_probe_launch(int* p) { dl_cuid_probe_kernel<<<1,1>>>(p); }
")

execute_process(
  COMMAND ${DL_CLANG} -x hip --offload-host-only ${DL_OFFLOAD_ARCH_FLAGS}
          ${DL_HIP_COMPILER_FLAGS} -cuid=${DL_SHARED_CUID} -std=c++17
          -c -o "${_cuid_probe_obj}" "${_cuid_probe_src}"
  RESULT_VARIABLE _cuid_probe_result
  OUTPUT_VARIABLE _cuid_probe_stdout
  ERROR_VARIABLE _cuid_probe_stderr
)
if(NOT _cuid_probe_result EQUAL 0)
  message(FATAL_ERROR "Device Linker: CUID probe compile failed:\n${_cuid_probe_stdout}\n${_cuid_probe_stderr}")
endif()

execute_process(
  COMMAND ${DL_READELF} -sW "${_cuid_probe_obj}"
  OUTPUT_VARIABLE _cuid_probe_syms
)
string(REGEX MATCH "__hip_fatbin_([0-9a-f]+)" _fatbin_match "${_cuid_probe_syms}")
if(NOT CMAKE_MATCH_1)
  message(FATAL_ERROR "Device Linker: could not discover __hip_fatbin_<hash> symbol from CUID probe. readelf output:\n${_cuid_probe_syms}")
endif()
set(DL_FATBIN_HASH "${CMAKE_MATCH_1}")
set(DL_FATBIN_SYMBOL "__hip_fatbin_${DL_FATBIN_HASH}")
set(DL_CUID_SYMBOL "__hip_cuid_${DL_FATBIN_HASH}")
message(STATUS "Device Linker: shared CUID '${DL_SHARED_CUID}' -> ${DL_FATBIN_SYMBOL}")

# ===========================================================================
# Mixed host+device TU registration
#
# Each registered TU gets, further below:
#   - (if HAS_DEVICE) a per-arch device-only object via the driver's
#     --emit-device-obj mode, containing ALL of its __global__ kernels
#     (no extraction -- these are standalone entries with their own .kd, not
#     participants in the specialized-kernel dispatcher's resource
#     aggregation), merged into that arch's device.elf alongside the
#     specialized kernels (see the per-arch loop below).
#   - A host-only object (--offload-host-only -cuid=<shared>), referencing
#     rather than embedding the fatbin, then localized (see the "Mixed-TU
#     host compiles" loop after glue.o) and added to DEVICE_LINKER_OBJECTS.
#
# `common` (common.cu.cpp) is HAS_DEVICE FALSE: it has no __global__ kernels
# of its own -- the driver's --link mode already compiles+patches+links its
# generic dispatcher kernel directly into every arch's device.elf as part of
# the existing resource-aggregation pipeline (see --dispatcher= below). It
# only needs the host-side registration change here: previously the one TU
# that explicitly embedded DEVICE_HIPFB via -fcuda-include-gpubinary, now
# just another shared-cuid reference like the rest (glue.o is the sole
# embedder).
# ===========================================================================
set(DL_MIXED_TU_NAMES "")

macro(dl_register_mixed_tu NAME SRC)
  cmake_parse_arguments(MTU "HAS_DEVICE;SUPPRESS_WARNINGS;INHERIT_FLAGS;DEPFILE_TRACKING" "" "" ${ARGN})
  list(APPEND DL_MIXED_TU_NAMES "${NAME}")
  set(DL_MTU_SRC_${NAME} "${SRC}")
  set(DL_MTU_HAS_DEVICE_${NAME} ${MTU_HAS_DEVICE})
  set(DL_MTU_SUPPRESS_WARNINGS_${NAME} ${MTU_SUPPRESS_WARNINGS})
  set(DL_MTU_INHERIT_FLAGS_${NAME} ${MTU_INHERIT_FLAGS})
  set(DL_MTU_DEPFILE_TRACKING_${NAME} ${MTU_DEPFILE_TRACKING})
endmacro()

dl_register_mixed_tu(common "${HIPIFY_DIR}/src/device/common.cu.cpp"
  INHERIT_FLAGS)
dl_register_mixed_tu(onerank "${HIPIFY_DIR}/src/device/onerank.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS)
# Dependency tracking note (CMake >= 3.20 vs < 3.20): add_custom_command only
# rebuilds when files listed in DEPENDS change. It does NOT automatically
# track transitive headers (e.g. ce_coll.h), so a struct layout change in a
# header would not invalidate collectives.o. CMake 3.20 DEPFILE support fixes
# this by reading the compiler-generated .d file; on older CMake the
# workaround is: touch hipify/src/collectives.cc.
dl_register_mixed_tu(collectives "${HIPIFY_DIR}/src/collectives.cc"
  HAS_DEVICE INHERIT_FLAGS DEPFILE_TRACKING)
# ce_reduce.cc itself is now pure host code (a dispatcher calling into
# per-instantiation launchers; see src/device/ce_reduce/generate.py) and is
# compiled normally as part of the main rccl target -- NOT registered here.
# Its 40 (type, redop) __global__ kernel instantiations live in
# gensrc/ce_reduce/*.cpp instead (one per instantiation, so ninja
# parallelizes them like every other device TU -- two of the 40,
# int8_t/uint8_t Min/Max, individually generate ~56K instructions and used
# to dominate the whole build's wall-clock time when they all lived in one
# TU). Globbed dynamically, mirroring the sym_* pattern below.
file(GLOB _ce_reduce_srcs CONFIGURE_DEPENDS "${HIPIFY_DIR}/gensrc/ce_reduce/*.cpp")
foreach(_ce_reduce_src IN LISTS _ce_reduce_srcs)
  get_filename_component(_ce_reduce_name "${_ce_reduce_src}" NAME_WE)
  string(MAKE_C_IDENTIFIER "${_ce_reduce_name}" _ce_reduce_name_id)
  dl_register_mixed_tu(ce_reduce_${_ce_reduce_name_id} "${_ce_reduce_src}" HAS_DEVICE SUPPRESS_WARNINGS)
endforeach()
dl_register_mixed_tu(dda_all_reduce_ipc "${HIPIFY_DIR}/src/dda_all_reduce_ipc.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS)
dl_register_mixed_tu(dda_reduce_scatter_ipc "${HIPIFY_DIR}/src/dda_reduce_scatter_ipc.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS)
dl_register_mixed_tu(dda_all_gather_ipc "${HIPIFY_DIR}/src/dda_all_gather_ipc.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS)
dl_register_mixed_tu(dda_alltoall_ipc "${HIPIFY_DIR}/src/dda_alltoall_ipc.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS)
dl_register_mixed_tu(dda_all_reduce_fabric "${HIPIFY_DIR}/src/dda_all_reduce_fabric.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_all_reduce_fabric_ll "${HIPIFY_DIR}/src/dda_all_reduce_fabric_ll.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_all_reduce_fabric_ll128 "${HIPIFY_DIR}/src/dda_all_reduce_fabric_ll128.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_reduce_scatter_fabric "${HIPIFY_DIR}/src/dda_reduce_scatter_fabric.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_all_gather_fabric "${HIPIFY_DIR}/src/dda_all_gather_fabric.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_all_gather_fabric_ll "${HIPIFY_DIR}/src/dda_all_gather_fabric_ll.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_all_gather_fabric_ll128 "${HIPIFY_DIR}/src/dda_all_gather_fabric_ll128.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_alltoall_fabric "${HIPIFY_DIR}/src/dda_alltoall_fabric.cu.cpp"
  HAS_DEVICE INHERIT_FLAGS SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_alltoall_fabric_ll "${HIPIFY_DIR}/src/dda_alltoall_fabric_ll.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_alltoall_fabric_ll128 "${HIPIFY_DIR}/src/dda_alltoall_fabric_ll128.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_reduce_scatter_fabric_ll "${HIPIFY_DIR}/src/dda_reduce_scatter_fabric_ll.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)
dl_register_mixed_tu(dda_reduce_scatter_fabric_ll128 "${HIPIFY_DIR}/src/dda_reduce_scatter_fabric_ll128.cu.cpp"
  HAS_DEVICE SUPPRESS_WARNINGS)

# Symmetric kernels: per-instantiation device TUs from gensrc/symmetric/.
# Each instantiation file defines a handful of __global__ ncclSymkDevKernel_*
# entries. SYM names are globbed dynamically (vs the fixed names above)
# because the symmetric generator emits one TU per instantiation.
if(GENERATE_SYM_KERNELS)
  file(GLOB _sym_srcs CONFIGURE_DEPENDS "${HIPIFY_DIR}/gensrc/symmetric/*.cpp")
  foreach(_sym_src IN LISTS _sym_srcs)
    get_filename_component(_sym_name "${_sym_src}" NAME_WE)
    string(MAKE_C_IDENTIFIER "${_sym_name}" _sym_name_id)
    dl_register_mixed_tu(sym_${_sym_name_id} "${_sym_src}" HAS_DEVICE INHERIT_FLAGS)
  endforeach()
endif()

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
  )
  target_compile_definitions(${_dev_target} PRIVATE RCCL_DEVICE_LINKER)
  target_link_libraries(${_dev_target} PRIVATE rccl_device_defs)

  add_dependencies(${_dev_target} hipify_all copy_nccl_device_headers)
  if(ENABLE_ROCSHMEM AND TARGET rocshmem_static)
    # rocSHMEM headers land in ext/rocshmem/include only after ExternalProject
    # completes; ensure they are installed before device kernels start compiling.
    add_dependencies(${_dev_target} rocshmem_static)
  endif()

  # =========================================================================
  # Mixed-TU per-arch device objects: merged into this arch's device.elf
  # alongside the specialized kernels above (see dl_register_mixed_tu doc).
  # =========================================================================
  set(_mixed_dev_objs "")
  foreach(_mtu_name IN LISTS DL_MIXED_TU_NAMES)
    if(NOT DL_MTU_HAS_DEVICE_${_mtu_name})
      continue()
    endif()
    set(_mtu_dev_obj "${DL_ARCH_DIR}/${_mtu_name}.o")
    add_custom_command(
      OUTPUT  ${_mtu_dev_obj}
      COMMAND ${CMAKE_RCCLDEV_COMPILER}
        --emit-device-obj
        --arch=${DL_GPU_TARGET}
        --clang=${DL_CLANG}
        ${DL_HIP_COMPILER_FLAGS}
        -DRCCL_DEVICE_LINKER
        ${_link_def_flags}
        ${_host_inc_flags}
        ${DL_OPT_FLAGS}
        -std=c++17
        -o ${_mtu_dev_obj}
        ${DL_MTU_SRC_${_mtu_name}}
      DEPENDS ${DL_MTU_SRC_${_mtu_name}}
      COMMENT "DL [${DL_GPU_TARGET}] device-obj: ${_mtu_name}"
      VERBATIM
      COMMAND_EXPAND_LISTS
    )
    list(APPEND _mixed_dev_objs "${_mtu_dev_obj}")
  endforeach()

  # =========================================================================
  # Link step: driver --link mode produces device.elf
  # =========================================================================
  set(ARCH_DEVICE_ELF "${DL_ARCH_DIR}/device.elf")

  set(_link_rsp "${DL_ARCH_DIR}/link_objects.rsp")
  list(JOIN _mixed_dev_objs "\n" _mixed_dev_objs_joined)
  file(GENERATE OUTPUT "${_link_rsp}"
    CONTENT "$<JOIN:$<TARGET_OBJECTS:${_dev_target}>,\n>\n${_mixed_dev_objs_joined}\n")

  # When rocSHMEM is enabled, pass the per-arch device bitcode to the driver.
  # rocSHMEM device API symbols have hidden visibility and must be statically
  # present in the device ELF — they cannot be imported from a shared library.
  set(_rocshmem_bitcode_arg "")
  set(_rocshmem_link_depends "")
  if(ENABLE_ROCSHMEM AND ROCSHMEM_INSTALL_DIR)
    set(_rocshmem_bc "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem_device_${DL_GPU_TARGET}.bc")
    set(_rocshmem_bitcode_arg "--rocshmem-bitcode=${_rocshmem_bc}")
    # Do NOT add _rocshmem_bc to DEPENDS: rocSHMEM only supports a subset of
    # GPU_TARGETS (e.g. gfx90a, gfx942, gfx950) and the bitcode files don't
    # exist at cmake configure time (ExternalProject).  The Python driver checks
    # existence at build time and skips silently for unsupported arches.
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
    DEPENDS ${_dev_target} ${_mixed_dev_objs} ${HIPIFY_DIR}/src/device/common.cu.cpp ${_rocshmem_link_depends}
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
# (now the ONE unified image: specialized dispatcher kernels + every mixed
# TU's kernels, for every arch)
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
# Glue object: the ONE place that embeds DEVICE_HIPFB's bytes, defining
# DL_FATBIN_SYMBOL globally via .incbin. Every mixed TU (including common)
# only references this symbol (undefined in their own objects); the final
# librccl.so link resolves all of them against this single definition. glue.s
# itself is static (the symbol name and DEVICE_HIPFB's path are both already
# known at configure time); glue.o still depends on DEVICE_HIPFB so it's
# reassembled -- re-running .incbin -- whenever the bundled image's bytes
# change.
# ===========================================================================
set(DL_GLUE_S "${DEVICE_BUILD_DIR}/glue.s")
file(WRITE "${DL_GLUE_S}"
"\t.section .hip_fatbin, \"a\", @progbits
\t.globl ${DL_FATBIN_SYMBOL}
\t.p2align 12
${DL_FATBIN_SYMBOL}:
\t.incbin \"${DEVICE_HIPFB}\"
")
set(DL_GLUE_OBJ "${DEVICE_BUILD_DIR}/glue.o")
add_custom_command(
  OUTPUT  ${DL_GLUE_OBJ}
  COMMAND ${DL_CLANG} -c -o ${DL_GLUE_OBJ} ${DL_GLUE_S}
  DEPENDS ${DEVICE_HIPFB} ${DL_GLUE_S}
  COMMENT "DL glue: single embed of ${DL_FATBIN_SYMBOL}"
  VERBATIM
)

# ===========================================================================
# Mixed-TU host compiles: --offload-host-only -cuid=<shared>, referencing
# (not embedding) the fatbin, then llvm-objcopy --localize-symbol strips the
# per-TU __hip_cuid_<hash> duplicate-CUID marker so linking them all
# together doesn't hit a duplicate-symbol error (see the "Shared CUID"
# section above for why that marker exists and why we deliberately want to
# suppress its complaint here).
# ===========================================================================
set(DEVICE_LINKER_OBJECTS "${DL_GLUE_OBJ}")

foreach(_mtu_name IN LISTS DL_MIXED_TU_NAMES)
  set(_mtu_raw_obj   "${DEVICE_BUILD_DIR}/${_mtu_name}_raw.o")
  set(_mtu_final_obj "${DEVICE_BUILD_DIR}/${_mtu_name}.o")

  set(_mtu_w_flag "")
  if(DL_MTU_SUPPRESS_WARNINGS_${_mtu_name})
    set(_mtu_w_flag -w)
  endif()
  set(_mtu_inherit_flags "")
  if(DL_MTU_INHERIT_FLAGS_${_mtu_name})
    set(_mtu_inherit_flags ${DL_INHERITED_FLAGS})
  endif()

  set(_mtu_mdmf_flags "")
  set(_mtu_depfile_kw "")
  if(DL_MTU_DEPFILE_TRACKING_${_mtu_name} AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
    set(_mtu_depfile "${DEVICE_BUILD_DIR}/${_mtu_name}.d")
    set(_mtu_mdmf_flags -MD -MF ${_mtu_depfile})
    set(_mtu_depfile_kw DEPFILE ${_mtu_depfile})
  endif()

  add_custom_command(
    OUTPUT  ${_mtu_raw_obj}
    COMMAND ${DL_CLANG}
      -x hip --offload-host-only ${DL_OFFLOAD_ARCH_FLAGS}
      -cuid=${DL_SHARED_CUID}
      ${DL_HIP_COMPILER_FLAGS}
      -DRCCL_DEVICE_LINKER
      ${_link_def_flags}
      ${_host_inc_flags}
      ${DL_OPT_FLAGS}
      ${_mtu_inherit_flags}
      -std=c++17
      -fPIC
      ${_mtu_w_flag}
      ${_mtu_mdmf_flags}
      -c -o ${_mtu_raw_obj}
      ${DL_MTU_SRC_${_mtu_name}}
    DEPENDS ${DL_MTU_SRC_${_mtu_name}}
    ${_mtu_depfile_kw}
    COMMENT "DL compile (host, shared cuid): ${_mtu_name}"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )

  add_custom_command(
    OUTPUT  ${_mtu_final_obj}
    COMMAND ${DL_OBJCOPY} --localize-symbol=${DL_CUID_SYMBOL} ${_mtu_raw_obj} ${_mtu_final_obj}
    DEPENDS ${_mtu_raw_obj}
    COMMENT "DL localize CUID marker: ${_mtu_name}"
    VERBATIM
  )

  list(APPEND DEVICE_LINKER_OBJECTS "${_mtu_final_obj}")
endforeach()

# ===========================================================================
# Top-level target
# ===========================================================================
add_custom_target(device_linker_build ALL
  DEPENDS ${DEVICE_LINKER_OBJECTS}
)
add_dependencies(device_linker_build hipify_all copy_nccl_device_headers)

# ===========================================================================
# Optional: emit LLVM IR (ninja device_ir)
# ===========================================================================
add_custom_target(device_ir DEPENDS ${ALL_IR_FILES})
add_dependencies(device_ir hipify_all copy_nccl_device_headers)
