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

# Device coverage requires the same ROCm 7.15+ compiler selected for the
# main build, including its packaged device profile runtime.
get_filename_component(_dl_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
if(ENABLE_FULL_COVERAGE)
  set(DL_CLANG "${CMAKE_CXX_COMPILER}")
  message(STATUS "Device Linker: DL_CLANG pinned to CMAKE_CXX_COMPILER for device coverage = ${DL_CLANG}")
else()
  find_program(DL_CLANG NAMES amdclang++ clang++
    HINTS "${_dl_compiler_dir}" "${ROCM_PATH}/bin" REQUIRED)
endif()
# Pick clang-offload-bundler from the SAME toolchain as DL_CLANG: a bundler from a
# different LLVM build can load a foreign libclang-cpp.so and crash at startup.
get_filename_component(_dl_clang_dir "${DL_CLANG}" REALPATH)
get_filename_component(_dl_clang_dir "${_dl_clang_dir}" DIRECTORY)
unset(DL_BUNDLER CACHE)
unset(DL_BUNDLER)
find_program(DL_BUNDLER NAMES clang-offload-bundler
  HINTS "${_dl_clang_dir}" "${_dl_compiler_dir}/../lib/llvm/bin"
        "${ROCM_PATH}/llvm/bin")
if(NOT DL_BUNDLER)
  message(FATAL_ERROR
    "Device Linker: clang-offload-bundler was not found next to the selected "
    "compiler '${DL_CLANG}' (searched '${_dl_clang_dir}', "
    "'${_dl_compiler_dir}/../lib/llvm/bin', '${ROCM_PATH}/llvm/bin').")
endif()
message(STATUS "Device Linker: clang-offload-bundler = ${DL_BUNDLER}")

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

# The --link step is the one consumer that cannot take a -fvisibility= option:
# the ncclDevFunc_* device functions have to stay visible, or the device link
# fails with "recompile with -fPIC". The host objects below still need it.
set(DL_LINK_INHERITED_FLAGS "")
foreach(_opt IN LISTS DL_INHERITED_FLAGS)
  if(_opt MATCHES "^-fvisibility=")
    continue()
  endif()
  list(APPEND DL_LINK_INHERITED_FLAGS "${_opt}")
endforeach()

# ---------------------------------------------------------------------------
# Parse GPU_TARGETS into a bare processor name plus the full target ID.
# ---------------------------------------------------------------------------
include(${CMAKE_CURRENT_LIST_DIR}/DeviceLinkerTargets.cmake)
dl_parse_gpu_targets(TARGETS ${GPU_TARGETS}
  BARE_VAR DL_GPU_TARGETS
  FLAGS_VAR DL_OFFLOAD_ARCH_FLAGS
  ID_PREFIX DL_TARGET_ID_)
message(STATUS "Device Linker: GPU targets = ${GPU_TARGETS}")

# ---------------------------------------------------------------------------
# Optimization flags (passed to both compile and link modes of the driver)
# ---------------------------------------------------------------------------
if(ENABLE_FULL_COVERAGE)
  # Drop DWARF (-g0) even under --debug. LLVM source-based coverage lives in
  # __llvm_covfun/__llvm_covmap + __llvm_prf_*, not DWARF, so llvm-cov needs
  # only device.elf + .profdata. With thousands of specialized kernels, keeping
  # debug info pushes the per-arch device.elf past the ~2 GB (INT32_MAX) HIP
  # fat-binary unbundling limit, after which every kernel launch fails.
  set(DL_OPT_FLAGS -O1 -g0)
elseif(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(DL_OPT_FLAGS -O1 -g)
else()
  set(DL_OPT_FLAGS -O3)
endif()

# ---------------------------------------------------------------------------
# Device-side coverage instrumentation (ENABLE_FULL_COVERAGE).
#
# We thread -fprofile-instr-generate / -fcoverage-mapping through every
# device compile in this pipeline (per-kernel RCCLDEV compiles, dispatcher
# compile inside --link, and the standalone fat-object compiles below).
# The assembly-extract pass passes them through to the underlying clang
# invocation; the LLVM profile metadata sections (__llvm_prf_*, __llvm_cov*)
# survive extraction because the extractor only strips the amdhsa_kernel /
# amdgpu_metadata ranges.
#
# For the per-arch device.elf to link cleanly we must also resolve the
# `__llvm_profile_runtime` reference emitted by each instrumented TU. We
# locate the per-arch device profile runtime archive (libclang_rt.profile.a
# under the amdgcn-amd-amdhsa resource sub-directory) and pass it to the
# rccl-device-compile driver, which forwards it to ld.lld.
# ---------------------------------------------------------------------------
set(DL_COVERAGE_FLAGS "")
set(DL_DEVICE_PROFILE_RT "")
if(ENABLE_FULL_COVERAGE)
  set(DL_COVERAGE_FLAGS -fprofile-instr-generate -fcoverage-mapping)
  set(DL_DEVICE_PROFILE_RT "${RCCL_DEVICE_PROFILE_RT}")
  if(NOT EXISTS "${DL_DEVICE_PROFILE_RT}")
    message(FATAL_ERROR
      "Device Linker: the preflighted device profile runtime is missing: "
      "'${DL_DEVICE_PROFILE_RT}'. Reconfigure the build.")
  endif()
  message(STATUS
    "Device Linker: coverage enabled, device profile RT = ${DL_DEVICE_PROFILE_RT}")
endif()

# Profile section anchor (ENABLE_FULL_COVERAGE).
#
# A coverage-instrumented device TU that emits no counters (e.g. a fat object
# with no device kernels, like sym_reduce_scatter.o) still emits an
# __llvm_profile_sections descriptor referencing __start/__stop___llvm_prf_{cnts,data}.
# With no __llvm_prf_cnts/__llvm_prf_data input section, the linker cannot synthesize
# those boundary symbols and the device link fails with "undefined hidden symbol:
# __start___llvm_prf_cnts". We link a tiny anchor that defines empty, retained
# __llvm_prf_{cnts,data} sections so the boundary symbols are always defined; when a
# TU does emit counters the empty sections merge in as a zero-byte no-op. The anchor
# is generic amdgcn bitcode (no mcpu) so every per-arch device link LTO-compiles it.
# It goes on each per-arch link line with the mixed-TU device objects.
set(DL_DEVICE_PROFILE_ANCHOR "")
if(DL_DEVICE_PROFILE_RT)
  file(MAKE_DIRECTORY "${DEVICE_BUILD_DIR}")
  set(_dl_anchor_src "${DEVICE_BUILD_DIR}/prf_section_anchor.c")
  set(_dl_anchor_bc  "${DEVICE_BUILD_DIR}/prf_section_anchor.bc")
  file(WRITE "${_dl_anchor_src}"
"/* Auto-generated by cmake/DeviceLinker.cmake for ENABLE_FULL_COVERAGE.\n"
"   Empty, retained profile counter/data sections so the device linker always\n"
"   defines __start/__stop___llvm_prf_{cnts,data}, even for instrumented TUs that\n"
"   reference those boundaries but emit no counters (e.g. no device kernels). */\n"
"__attribute__((used, retain, section(\"__llvm_prf_cnts\")))\n"
"static char __rccl_prf_cnts_anchor[0];\n"
"__attribute__((used, retain, section(\"__llvm_prf_data\")))\n"
"static char __rccl_prf_data_anchor[0];\n")
  execute_process(
    COMMAND ${DL_CLANG} --target=amdgcn-amd-amdhsa -c -emit-llvm -O1
            -o "${_dl_anchor_bc}" "${_dl_anchor_src}"
    RESULT_VARIABLE _dl_anchor_rc
    ERROR_VARIABLE _dl_anchor_err)
  if(NOT _dl_anchor_rc EQUAL 0 OR NOT EXISTS "${_dl_anchor_bc}")
    message(FATAL_ERROR
      "Device Linker: failed to build profile section anchor bitcode.\n${_dl_anchor_err}")
  endif()
  set(DL_DEVICE_PROFILE_ANCHOR "${_dl_anchor_bc}")
  message(STATUS "Device Linker: profile section anchor = ${DL_DEVICE_PROFILE_ANCHOR}")
endif()

# ---------------------------------------------------------------------------
# Main-module coverage drain: deterministic compilation-unit ID (-cuid).
#
# The device profile runtime drains a per-arch device.elf by looking up a host
# "shadow" variable __llvm_profile_sections_<CUIDHash> (registered from the host
# TU via __hipRegisterVar) and resolving it to the identically-named device
# global with hipGetSymbolAddress. The device global points at the module-wide
# __start/__stop___llvm_prf_{cnts,data,names} boundaries, so draining through
# ANY ONE descriptor drains the entire linked device.elf.
#
# The main RDC module is compiled in two separate steps: the device side
# (dispatcher common.cu.cpp inside --link, plus the per-kernel TUs) and the host
# side (common.cu.cpp --offload-host-only). With the default -fuse-cuid=hash each
# step derives a different CUID from its own file path + command line, so the
# host shadow name (e.g. a2f3...) never matches any of the device descriptors and
# hipGetSymbolAddress fails -> device counters read back as zero.
#
# Pin BOTH the dispatcher device compile and the host common.cu.cpp compile to
# the same explicit -cuid so their __llvm_profile_sections_<CUIDHash> names match
# (CUIDHash = MD5(cuid), identical across separate invocations). This is the
# compiler's intended lever for split host/device compilation (see the codegen
# test clang/test/CodeGenHIP/offload-pgo-sections.hip, which passes -cuid=abc to
# both cc1 jobs). It must NOT be applied to the per-kernel device TUs: they are
# distinct compilation units and a shared CUID would collide on the (non-weak)
# __llvm_profile_sections_<CUIDHash> device global at link time.
set(DL_MAIN_MODULE_CUID_FLAGS "")
if(ENABLE_FULL_COVERAGE)
  set(DL_MAIN_MODULE_CUID_FLAGS -cuid=rccl_main_module)
  message(STATUS "Device Linker: main-module coverage CUID pinned = rccl_main_module")
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

# fmt headers, named directly rather than read off fmt::fmt-header-only: the
# target carries its include directory in a $<BUILD_INTERFACE:...> generator
# expression, which cannot be resolved at configure time.
target_include_directories(rccl_device_defs SYSTEM INTERFACE
  "${PROJECT_SOURCE_DIR}/external/fmt/include")

# The same, as flags for the compiles below that bypass the RCCLDEV language.
get_target_property(_iface_defs rccl_device_defs INTERFACE_COMPILE_DEFINITIONS)
list(TRANSFORM _iface_defs PREPEND "-D")
set(_link_def_flags -DRCCL_DEVICE_LINKER ${_iface_defs})
list(REMOVE_DUPLICATES _link_def_flags)

get_target_property(_dev_includes rccl_device_defs INTERFACE_INCLUDE_DIRECTORIES)
list(TRANSFORM _dev_includes PREPEND "-I" OUTPUT_VARIABLE _link_inc_flags)
get_target_property(_dev_sys_includes rccl_device_defs INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
foreach(_inc ${_dev_sys_includes})
  if(NOT _inc IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    list(APPEND _link_inc_flags "-isystem${_inc}")
  endif()
endforeach()

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
  string(REGEX MATCHALL "__gfx[0-9a-z_]+__" _guard_archs "${GUARD}")
  string(REPLACE "-" "_" _gpu_macro "${GPU_TARGET}")
  if(NOT _guard_archs)
    set(${RESULT_VAR} TRUE PARENT_SCOPE)
    return()
  endif()
  foreach(_ga ${_guard_archs})
    string(REGEX REPLACE "^__(.+)__$" "\\1" _arch "${_ga}")
    if("${_arch}" STREQUAL "${_gpu_macro}")
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
# Mixed host+device TUs: sources outside the specialized-kernel pipeline that
# define kernels of their own. The device half is compiled once per GPU target
# and linked into that target's device.elf; the host half goes into
# DEVICE_LINKER_OBJECTS. Both halves are -fgpu-rdc with the same -cuid, so
# internal-linkage kernels and variables get the same name on each side, and
# the host half carries offload entries instead of an image of its own.
# ===========================================================================
set(DEVICE_LINKER_OBJECTS "")

# dl_add_fat_object(<name> <source> [SUPPRESS_WARNINGS] [DEPFILE_TRACKING]
#                   [DEFINES ...] [EXTRA_FLAGS ...] [EXTRA_DEPENDS ...])
# EXTRA_FLAGS and EXTRA_DEPENDS apply to the device half only.
function(dl_add_fat_object NAME SRC)
  cmake_parse_arguments(FO "SUPPRESS_WARNINGS;DEPFILE_TRACKING" ""
    "DEFINES;EXTRA_FLAGS;EXTRA_DEPENDS" ${ARGN})
  if(FO_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "dl_add_fat_object(${NAME}): unknown arguments: ${FO_UNPARSED_ARGUMENTS}")
  endif()
  list(TRANSFORM FO_DEFINES PREPEND "-D")
  set(_flags ${DL_HIP_COMPILER_FLAGS} --offload-new-driver -fgpu-rdc -cuid=${NAME}
    ${FO_DEFINES} ${_link_def_flags} ${_link_inc_flags} ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS} ${DL_COVERAGE_FLAGS} -std=c++17 -fPIC)
  if(FO_SUPPRESS_WARNINGS)
    list(APPEND _flags -w)
  endif()
  foreach(_half host ${DL_GPU_TARGETS})
    if(_half STREQUAL "host")
      set(_obj "${DEVICE_BUILD_DIR}/${NAME}.o")
      set(_half_flags --offload-host-only ${DL_OFFLOAD_ARCH_FLAGS})
      set(_deps "")
    else()
      set(_obj "${DEVICE_BUILD_DIR}/${_half}/${NAME}.o")
      file(MAKE_DIRECTORY "${DEVICE_BUILD_DIR}/${_half}")
      # Each TU is self-contained on the device; internalize as non-RDC does.
      set(_half_flags --offload-device-only --offload-arch=${DL_TARGET_ID_${_half}}
        -fno-offload-lto -mllvm -amdgpu-internalize-symbols ${FO_EXTRA_FLAGS})
      set(_deps ${FO_EXTRA_DEPENDS})
      set(DL_MIXED_DEVICE_OBJS_${_half} ${DL_MIXED_DEVICE_OBJS_${_half}} "${_obj}" PARENT_SCOPE)
    endif()
    # Only <source> is a DEPENDS, so without a depfile a header edit leaves the
    # object stale. CMake < 3.20 has no DEPFILE for custom commands.
    set(_depfile "")
    if(FO_DEPFILE_TRACKING AND CMAKE_VERSION VERSION_GREATER_EQUAL "3.20")
      list(APPEND _half_flags -MD -MF "${_obj}.d")
      set(_depfile DEPFILE "${_obj}.d")
    endif()
    set(_commands COMMAND ${DL_CLANG} -x hip ${_half_flags} ${_flags} -c -o ${_obj} ${SRC})
    if(BUILD_DEVICE_ADDRESS_SANITIZER AND NOT _half STREQUAL "host")
      # See ir_codegen_cmd in tools/rccl-device-compile.
      set(_mt "")
      if(_depfile)
        set(_mt -MT ${_obj})
      endif()
      set(_commands
        COMMAND ${DL_CLANG} -x hip ${_half_flags} ${_mt} ${_flags} -emit-llvm -c -o ${_obj}.bc ${SRC}
        COMMAND ${CMAKE_RCCLDEV_COMPILER} --ir-codegen --arch=${_half}
          --target-id=${DL_TARGET_ID_${_half}} --clang=${DL_CLANG}
          --dasan-tu-rt=${RCCL_DASAN_TU_RT} ${_flags} -o ${_obj} ${_obj}.bc)
    endif()
    add_custom_command(
      OUTPUT  ${_obj}
      ${_commands}
      DEPENDS ${SRC} ${_deps}
      ${_depfile}
      COMMENT "DL compile (${_half}): ${NAME}"
      VERBATIM
    )
  endforeach()
  set(DEVICE_LINKER_OBJECTS ${DEVICE_LINKER_OBJECTS} "${DEVICE_BUILD_DIR}/${NAME}.o" PARENT_SCOPE)
endfunction()

dl_add_fat_object(onerank ${HIPIFY_DIR}/src/device/onerank.cu.cpp)
dl_add_fat_object(collectives ${HIPIFY_DIR}/src/collectives.cc DEPFILE_TRACKING)

# One TU per (type, redop); src/device/ce_reduce/generate.py says why.
file(GLOB _ce_reduce_srcs CONFIGURE_DEPENDS "${HIPIFY_DIR}/gensrc/ce_reduce/*.cpp")
foreach(_src IN LISTS _ce_reduce_srcs)
  get_filename_component(_name "${_src}" NAME_WE)
  dl_add_fat_object(${_name} ${_src} SUPPRESS_WARNINGS)
endforeach()

dl_add_fat_object(diagnostics_p2p ${HIPIFY_DIR}/src/diagnostics/device/p2p.cu.cpp)

set(_dda_dir ${HIPIFY_DIR}/src/algorithms/dda)
foreach(_coll all_reduce reduce_scatter all_gather alltoall)
  dl_add_fat_object(dda_${_coll}_ipc ${_dda_dir}/${_coll}/dda_${_coll}_ipc.cu.cpp)
endforeach()
foreach(_coll all_reduce reduce_scatter all_gather alltoall)
  foreach(_proto "" _ll _ll128)
    dl_add_fat_object(dda_${_coll}_fabric${_proto}
      ${_dda_dir}/${_coll}/dda_${_coll}_fabric${_proto}.cu.cpp SUPPRESS_WARNINGS)
  endforeach()
endforeach()

# ===========================================================================
# Symmetric kernels: per-instantiation device TUs from gensrc/symmetric/.
# Each instantiation file defines a handful of __global__ ncclSymkDevKernel_*
# entries.
# ===========================================================================
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
        -x hip --cuda-device-only --offload-arch=${DL_TARGET_ID_${_bc_arch}}
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
      COMMAND ${_llvm_link} ${_cm_bc} -o ${_qp_raw}
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
    # Only GIN symmetric kernels need the QP bitcode; non-GIN ones would
    # just ingest and DCE it, wasting compile time.
    set(_this_bc_flag "")
    set(_this_bc_deps "")
    if(_sym_name MATCHES "_gin")
      set(_this_bc_flag "${_sym_rocshmem_bc_flag}")
      set(_this_bc_deps "${_sym_rocshmem_deps}")
    endif()
    dl_add_fat_object(sym_${_sym_name} ${_sym_src}
      EXTRA_FLAGS ${_this_bc_flag} EXTRA_DEPENDS ${_this_bc_deps})
  endforeach()
endif()

# The GIN defines go here because these files are filtered out of the rccl
# target, so per-source properties never apply. Leaving only SDMA on keeps
# ncclGinCallImpl on its single backend branch. GDA must stay off, as these
# objects get no QP bitcode.
if(ENABLE_ROCSHMEM_GIN)
  foreach(_coll alltoall all_reduce)
    dl_add_fat_object(gin_${_coll}_sdma
      ${HIPIFY_DIR}/src/algorithms/gin/sdma/gin_${_coll}_sdma.cu.cpp
      DEFINES NCCL_GIN_ANVIL_SDMA_ENABLE=1 NCCL_GIN_PROXY_ENABLE=0
              NCCL_GIN_ROCSHMEM_GDA_ENABLE=0)
  endforeach()
  dl_add_fat_object(gin_anvil_conn_check_device
    ${HIPIFY_DIR}/src/gin/gin_anvil_conn_check_device.cc
    DEFINES ENABLE_ROCSHMEM_GIN)
endif()

# ===========================================================================
# Per-GPU-target: OBJECT library (compile) + link custom command
# ===========================================================================
set(ALL_DEVICE_ELFS "")
set(DL_BUNDLER_TARGETS "host-${_dl_host_triple}-")
set(DL_BUNDLER_INPUTS "--input=/dev/null")
set(ALL_IR_FILES "")

foreach(DL_GPU_TARGET ${DL_GPU_TARGETS})
  set(DL_TARGET_ID "${DL_TARGET_ID_${DL_GPU_TARGET}}")

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
    --target-id=${DL_TARGET_ID}
    --clang=${DL_CLANG}
    ${DL_OPT_FLAGS}
    ${DL_COVERAGE_FLAGS}
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
  if(BUILD_DEVICE_ADDRESS_SANITIZER)
    target_compile_options(${_dev_target} PRIVATE --dasan-tu-rt=${RCCL_DASAN_TU_RT})
  endif()
  target_compile_definitions(${_dev_target} PRIVATE RCCL_DEVICE_LINKER)
  target_link_libraries(${_dev_target} PRIVATE rccl_device_defs)

  add_dependencies(${_dev_target} hipify_all copy_nccl_device_headers)
  if((ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN) AND TARGET rocshmem_static)
    add_dependencies(${_dev_target} rocshmem_static)
  endif()
  if(ENABLE_ROCSHMEM_GIN AND TARGET copy_rocshmem_headers)
    add_dependencies(${_dev_target} copy_rocshmem_headers)
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

  set(_mixed_objs ${DL_MIXED_DEVICE_OBJS_${DL_GPU_TARGET}})
  if(DL_DEVICE_PROFILE_ANCHOR)
    list(APPEND _mixed_objs "${DL_DEVICE_PROFILE_ANCHOR}")
  endif()
  list(JOIN _mixed_objs "\n" _mixed_objs_lines)
  set(_link_rsp "${DL_ARCH_DIR}/link_objects.rsp")
  file(GENERATE OUTPUT "${_link_rsp}"
    CONTENT "$<JOIN:$<TARGET_OBJECTS:${_dev_target}>,\n>\n${_mixed_objs_lines}\n")

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

  set(_dl_profile_rt_arg "")
  if(DL_DEVICE_PROFILE_RT)
    set(_dl_profile_rt_arg "--profile-rt=${DL_DEVICE_PROFILE_RT}")
  endif()
  set(_dl_dasan_rt_arg "")
  if(BUILD_DEVICE_ADDRESS_SANITIZER)
    set(_dl_dasan_rt_arg "--dasan-rt=${RCCL_DASAN_DEVICE_RT}" "--dasan-tu-rt=${RCCL_DASAN_TU_RT}")
  endif()

  add_custom_command(
    OUTPUT  ${ARCH_DEVICE_ELF}
    COMMAND ${CMAKE_RCCLDEV_COMPILER}
      --link
      --arch=${DL_GPU_TARGET}
      --target-id=${DL_TARGET_ID}
      --clang=${DL_CLANG}
      ${DL_HIP_COMPILER_FLAGS}
      --dispatcher=${HIPIFY_DIR}/src/device/common.cu.cpp
      ${_rocshmem_bitcode_arg}
      ${_dl_profile_rt_arg}
      ${_dl_dasan_rt_arg}
      ${_link_def_flags}
      ${_link_inc_flags}
      ${DL_OPT_FLAGS}
      ${DL_LINK_INHERITED_FLAGS}
      ${DL_COVERAGE_FLAGS}
      ${DL_MAIN_MODULE_CUID_FLAGS}
      -std=c++17
      -o ${ARCH_DEVICE_ELF}
      @${_link_rsp}
    DEPENDS ${_dev_target} ${HIPIFY_DIR}/src/device/common.cu.cpp ${_mixed_objs} ${_rocshmem_link_depends}
    COMMENT "DL [${DL_GPU_TARGET}] link: device.elf"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )

  list(APPEND ALL_DEVICE_ELFS "${ARCH_DEVICE_ELF}")
  # The runtime matches this entry against the agent at load time, so it needs
  # the target ID, not the bare name.
  list(APPEND DL_BUNDLER_TARGETS "hip-amdgcn-amd-amdhsa--${DL_TARGET_ID}")
  list(APPEND DL_BUNDLER_INPUTS "--input=${ARCH_DEVICE_ELF}")

  # Friendly symlink next to librccl.so: device-${arch}.elf -> device_build/<dir>/device.elf
  # Lets coverage workflows (and inspection in general) reference the per-arch
  # device ELF without the leading-hyphen directory wart we use for CDNA build
  # scheduling. Same byte content that ends up packed into .hip_fatbin; llvm-cov
  # only needs this file + the merged .profdata for device-side reports.
  set(_dev_elf_link "${PROJECT_BINARY_DIR}/device-${DL_GPU_TARGET}.elf")
  add_custom_command(
    OUTPUT  ${_dev_elf_link}
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${ARCH_DEVICE_ELF} ${_dev_elf_link}
    DEPENDS ${ARCH_DEVICE_ELF}
    COMMENT "DL [${DL_GPU_TARGET}] symlink: device-${DL_GPU_TARGET}.elf -> device_build/.../device.elf"
    VERBATIM
  )
  list(APPEND DEVICE_ELF_SYMLINKS "${_dev_elf_link}")

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
        -x hip --offload-device-only --offload-arch=${DL_TARGET_ID}
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
  COMMENT "DL bundle: device.elf(s) -> device.hipfb [${GPU_TARGETS}]"
  VERBATIM
)

# ===========================================================================
# common.cu.cpp's host half, and the one registration of the device image for
# every kernel and variable in DEVICE_LINKER_OBJECTS' offload entries.
# ===========================================================================
set(COMMON_FAT_OBJ "${DEVICE_BUILD_DIR}/common.o")
add_custom_command(
  OUTPUT  ${COMMON_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip --offload-host-only ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    --offload-new-driver -fgpu-rdc
    ${_link_def_flags}
    ${_link_inc_flags}
    ${DL_OPT_FLAGS}
    ${DL_INHERITED_FLAGS}
    ${DL_COVERAGE_FLAGS}
    ${DL_MAIN_MODULE_CUID_FLAGS}
    -std=c++17
    -fPIC
    -c -o ${COMMON_FAT_OBJ}
    ${HIPIFY_DIR}/src/device/common.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/device/common.cu.cpp
  COMMENT "DL compile (host): common"
  VERBATIM
)

set(DL_IMAGE_OBJ "${DEVICE_BUILD_DIR}/device_image.o")
set(_image_src "${PROJECT_SOURCE_DIR}/src/device/device_image.cc")
add_custom_command(
  OUTPUT  ${DL_IMAGE_OBJ}
  COMMAND ${DL_CLANG} -x c++ -O2 -fPIC -std=c++17
    "-DRCCL_DEVICE_IMAGE=\"${DEVICE_HIPFB}\""
    -c -o ${DL_IMAGE_OBJ} ${_image_src}
  DEPENDS ${DEVICE_HIPFB} ${_image_src}
  COMMENT "DL compile: device image registration"
  VERBATIM
)
list(PREPEND DEVICE_LINKER_OBJECTS ${DL_IMAGE_OBJ} ${COMMON_FAT_OBJ})

# ===========================================================================
# Top-level target
# ===========================================================================
add_custom_target(device_linker_build ALL
  DEPENDS ${DEVICE_LINKER_OBJECTS} ${DEVICE_ELF_SYMLINKS}
)
add_dependencies(device_linker_build hipify_all copy_nccl_device_headers)
if(ENABLE_ROCSHMEM_GIN AND TARGET copy_rocshmem_headers)
  add_dependencies(device_linker_build copy_rocshmem_headers)
endif()
if((ENABLE_ROCSHMEM OR ENABLE_ROCSHMEM_GIN) AND TARGET rocshmem_static)
  # The fat objects above include GIN device headers, which pull in rocSHMEM
  # headers installed to ext/rocshmem/include by the ExternalProject.
  add_dependencies(device_linker_build rocshmem_static)
endif()

# ===========================================================================
# Optional: emit LLVM IR (ninja device_ir)
# ===========================================================================
add_custom_target(device_ir DEPENDS ${ALL_IR_FILES})
add_dependencies(device_ir hipify_all copy_nccl_device_headers)
if(ENABLE_ROCSHMEM_GIN AND TARGET copy_rocshmem_headers)
  add_dependencies(device_ir copy_rocshmem_headers)
endif()
