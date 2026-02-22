# cmake/SplitDeviceCompile.cmake
#
# Split device compilation pipeline for RCCL.
#
# =============================================================================
# Overview
# =============================================================================
#
# Instead of the standard -fgpu-rdc flow (which defers all backend passes to
# link time, serialising them), this module pre-compiles each device TU through
# the full LLVM backend independently and in parallel:
#
#   source.cpp -> LLVM bitcode (.bc) -> assembly (.s) -> device ELF (.o)
#                                                           |
#   source.cpp -> host stub (.host.o) ----------------------+
#                                                           v
#                                              fat object (.fat.o) via bundler
#
# The fat objects are then linked with `amdclang++ -fgpu-rdc --hip-link` which
# only performs lightweight device-side symbol resolution (~1-2 seconds) instead
# of re-running the entire backend.
#
# =============================================================================
# Kernel metadata patching for indirect function calls
# =============================================================================
#
# Problem
# -------
# RCCL uses indirect function calls (IFC) to dispatch collective operations at
# runtime.  The dispatch kernels (ncclDevKernel_Generic_{1,2,4}) in common.cu
# call into per-collective functions (ncclDevFunc_*) via function-pointer tables
# (ncclDevFuncTable_{1,2,4}).  Each collective function is compiled as its own
# translation unit with -DNCCL_FUNC_ONLY.
#
# When the AMDGPU backend compiles a kernel, it writes a binary kernel
# descriptor (KD) in the .rodata ELF section.  This KD contains hardware
# register allocation fields (COMPUTE_PGM_RSRC1/RSRC3) that tell the GPU's
# shader processor how many VGPRs, AGPRs, and SGPRs to reserve per wavefront.
# If a callee function uses more registers than the kernel descriptor declares,
# the hardware faults at dispatch time.
#
# In a normal (non-split) -fgpu-rdc build, the linker sees all TUs and can
# propagate callee resource usage back to the kernel descriptor.  In our split
# pipeline, each TU is compiled through the backend in isolation.  The kernel
# TU (common.cu) has no visibility into the callees' register pressure, so the
# AMDGPU backend records only the kernel's own register usage.
#
# Concretely, the AMDGPU backend emits assembly like:
#
#   .set amdgpu.max_num_vgpr, 54      # module-wide max across local functions
#   .set amdgpu.max_num_agpr, 0
#   .set amdgpu.max_num_sgpr, 34
#
#   .set KERNEL.num_vgpr, max(62, amdgpu.max_num_vgpr)   # = max(62,54) = 62
#   .set KERNEL.num_agpr, max(0,  amdgpu.max_num_agpr)   # = max(0, 0)  = 0
#
#   .amdhsa_next_free_vgpr  max(totalnumvgprs(KERNEL.num_agpr, KERNEL.num_vgpr), 1, 0)
#   .amdhsa_accum_offset    <expression of KERNEL.num_vgpr>
#   .amdhsa_next_free_sgpr  max(KERNEL.numbered_sgpr+6, 1, 0)-6
#
# The amdgpu.max_num_* symbols are the knobs: they represent the worst-case
# resource usage of any function reachable through indirect calls.  When all
# TUs are in one module these values are correct.  When the kernel is compiled
# alone, they only reflect common.cu's own helper functions.
#
# On gfx950 (MI350) this mismatch causes a hardware fault because the callee
# functions actually use significantly more registers:
#
#   Resource    Kernel declares    Callees need
#   --------    ---------------    ------------
#   VGPRs       62                 128  (v0-v127)
#   AGPRs        0                  64  (a0-a63)
#   SGPRs       88                 102  (s0-s101)
#
# Solution
# --------
# ALL device TUs compile through assembly (bc → .s → .o) rather than directly
# to object code (bc → .o).  The bc→asm step is the same cost as bc→obj
# (identical LLVM backend work, only the emission format differs).  The
# asm→obj step is just the assembler and completes in < 1 second per TU.
#
# For callee TUs, a metadata sidecar (.meta) is extracted via grep immediately
# after bc→asm, while the .s file is still in page cache.  The sidecar
# contains just the amdgpu.max_num_* lines (~4 lines, <200 bytes).
#
# For kernel TUs, the patch step reads all callee sidecar files (122 files ×
# ~200 bytes = ~24 KB total), finds the global maximum for each register
# class, and rewrites the three .set directives in the kernel assembly:
#
#   .set amdgpu.max_num_vgpr, 128   # was 54
#   .set amdgpu.max_num_agpr,  64   # was 0
#   .set amdgpu.max_num_sgpr, 102   # was 34
#
# The existing kernel-descriptor expressions automatically recompute the
# correct values.  For example, totalnumvgprs(64, 128) yields 192 total
# physical registers (128 arch VGPRs + 64 AGPRs in the unified file),
# and accum_offset becomes 128 so that AGPRs start at physical index 128.
#
# The patched assembly is then assembled to a device ELF.  The entire patch
# step (read sidecars + patch 1.3 MB assembly + write) takes < 0.2 seconds.
#
# Dependency graph:
#
#   callee bc→asm (parallel) ──→ callee .meta (grep) ──→ kernel patch ──→ kernel asm→obj
#                             └→ callee asm→obj (parallel, independent of kernel patch)
#
# The only serialization is the kernel patch waiting for all callee .meta
# files.  Since callee bc→asm is the same work as the original bc→obj, this
# adds no new serialization relative to the original pipeline -- only the
# near-instant patch step is inserted.
#
# Note on scratch / private segment
# ----------------------------------
# The kernel descriptor also has a private_segment_fixed_size field (scratch
# memory per work-item).  The kernel sets this to 0 because it does not itself
# spill to scratch.  Callee functions DO use scratch (up to ~1200 bytes
# cumulative), but this is covered by the uses_dynamic_stack flag which causes
# the HSA runtime to allocate a default stack budget (~8 KB) per work-item.
#
# Note on .note YAML metadata
# ----------------------------
# The .note section in the ELF contains a YAML copy of kernel metadata
# (.vgpr_count, .agpr_count, .sgpr_count).  These are hardcoded literals
# in the assembly.  The patch script updates them for IFC kernels
# (.uses_dynamic_stack: true) to max(kernel_value, callee_max), so that
# profiling and diagnostic tools report correct values.  The binary kernel
# descriptor (COMPUTE_PGM_RSRC1/RSRC3 in .rodata) is also patched via the
# .set amdgpu.max_num_* directives.
#

function(setup_split_device_compile)
  cmake_parse_arguments(SDC "" "TARGET;GPU_ARCH;ROCM_PATH;OUTPUT_DIR;BUNDLER_DEVICE_TARGET;BUNDLER_HOST_TARGET" "SOURCES;FUNC_ONLY_SOURCES;INCLUDE_DIRS;COMPILE_DEFS;COMPILE_OPTS" ${ARGN})

  if(NOT SDC_TARGET)
    message(FATAL_ERROR "setup_split_device_compile: TARGET is required")
  endif()
  if(NOT SDC_GPU_ARCH)
    message(FATAL_ERROR "setup_split_device_compile: GPU_ARCH is required")
  endif()
  if(NOT SDC_ROCM_PATH)
    message(FATAL_ERROR "setup_split_device_compile: ROCM_PATH is required")
  endif()
  if(NOT SDC_OUTPUT_DIR)
    message(FATAL_ERROR "setup_split_device_compile: OUTPUT_DIR is required")
  endif()
  if(NOT SDC_BUNDLER_DEVICE_TARGET)
    message(FATAL_ERROR "setup_split_device_compile: BUNDLER_DEVICE_TARGET is required")
  endif()
  if(NOT SDC_BUNDLER_HOST_TARGET)
    message(FATAL_ERROR "setup_split_device_compile: BUNDLER_HOST_TARGET is required")
  endif()

  set(BUNDLER "${SDC_ROCM_PATH}/llvm/bin/clang-offload-bundler")
  set(PATCH_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/patch_kernel_metadata.cmake")

  # Output directories
  set(BC_DIR   "${SDC_OUTPUT_DIR}/split_device/bc")
  set(ASM_DIR  "${SDC_OUTPUT_DIR}/split_device/asm")
  set(DEV_DIR  "${SDC_OUTPUT_DIR}/split_device/dev_obj")
  set(HOST_DIR "${SDC_OUTPUT_DIR}/split_device/host_obj")
  set(FAT_DIR  "${SDC_OUTPUT_DIR}/split_device/fat_obj")
  file(MAKE_DIRECTORY ${BC_DIR} ${ASM_DIR} ${DEV_DIR} ${HOST_DIR} ${FAT_DIR})

  # Build the include-directory flags list: -I/path1 -I/path2 ...
  set(_inc_flags "")
  foreach(_dir ${SDC_INCLUDE_DIRS})
    list(APPEND _inc_flags "-I${_dir}")
  endforeach()

  # Build the compile-definition flags list: -DFOO -DBAR=123 ...
  set(_def_flags "")
  foreach(_def ${SDC_COMPILE_DEFS})
    list(APPEND _def_flags "-D${_def}")
  endforeach()

  # Forward compile options from the target (e.g. -mllvm flags, -W flags).
  # Filter out options that conflict with the split pipeline's own flags
  # or are irrelevant for device-only / host-only compilation.
  set(_fwd_compile_opts "")
  set(_skip_next OFF)
  foreach(_opt ${SDC_COMPILE_OPTS})
    if(_skip_next)
      # This is the argument to a previous flag (e.g. the value after -mllvm)
      list(APPEND _fwd_compile_opts "${_opt}")
      set(_skip_next OFF)
    elseif(_opt MATCHES "^-parallel-jobs"
        OR _opt MATCHES "^--offload-compress"
        OR _opt MATCHES "^--offload-arch"
        OR _opt MATCHES "^-fvisibility"
        OR _opt MATCHES "^-fgpu-rdc"
        OR _opt MATCHES "^-x$"
        OR _opt MATCHES "^-std="
        OR _opt MATCHES "^-O[0-3s]$"
        OR _opt MATCHES "^-fPIC")
      # Skip: these are already set explicitly or irrelevant
    elseif(_opt STREQUAL "-mllvm")
      # -mllvm takes the next arg; forward both
      list(APPEND _fwd_compile_opts "${_opt}")
      set(_skip_next ON)
    else()
      list(APPEND _fwd_compile_opts "${_opt}")
    endif()
  endforeach()

  set(ALL_FAT_OBJECTS "")
  set(_callee_meta_files "")
  set(_kernel_fnames "")
  set(_kernel_asm_files "")
  set(_kernel_dev_objs "")

  list(LENGTH SDC_SOURCES _n_sources)
  message(STATUS "Split device compile: ${_n_sources} TUs for ${SDC_GPU_ARCH}")

  foreach(src ${SDC_SOURCES})
    get_filename_component(fname ${src} NAME_WE)

    # Determine if this source needs -DNCCL_FUNC_ONLY
    set(_extra_defs "")
    set(_is_func_only FALSE)
    list(FIND SDC_FUNC_ONLY_SOURCES "${src}" _func_only_idx)
    if(NOT _func_only_idx EQUAL -1)
      set(_extra_defs "-DNCCL_FUNC_ONLY")
      set(_is_func_only TRUE)
    endif()

    set(BC_FILE  "${BC_DIR}/${fname}.${SDC_GPU_ARCH}.bc")
    set(ASM_FILE "${ASM_DIR}/${fname}.${SDC_GPU_ARCH}.s")
    set(DEV_OBJ  "${DEV_DIR}/${fname}.${SDC_GPU_ARCH}.o")
    set(HOST_OBJ "${HOST_DIR}/${fname}.host.o")
    set(FAT_OBJ  "${FAT_DIR}/${fname}.fat.o")

    # -- Step A: Compile to LLVM bitcode (device only) ------------------------
    add_custom_command(
      OUTPUT  ${BC_FILE}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x hip -std=c++17
        -fgpu-rdc
        --offload-device-only
        --offload-arch=${SDC_GPU_ARCH}
        -emit-llvm -c -O3
        ${_inc_flags}
        ${_def_flags}
        ${_extra_defs}
        ${_fwd_compile_opts}
        -fvisibility=hidden
        -Wno-unused-function
        -Wno-format-nonliteral
        -o ${BC_FILE} ${src}
      DEPENDS   ${src}
      COMMENT   "SPLIT[bc]  ${fname}"
      VERBATIM
    )

    # -- Step B1: Compile bitcode to assembly ---------------------------------
    # All TUs go through assembly so that callee metadata (.set amdgpu.max_num_*)
    # is available as text for the kernel metadata patching step.  The asm→obj
    # step (B2) that follows is just the assembler and is very fast.
    add_custom_command(
      OUTPUT  ${ASM_FILE}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x ir
        -target amdgcn-amd-amdhsa
        -mcpu=${SDC_GPU_ARCH}
        -O3 -S
        -o ${ASM_FILE} ${BC_FILE}
      DEPENDS   ${BC_FILE}
      COMMENT   "SPLIT[asm] ${fname}"
      VERBATIM
    )

    if(_is_func_only)
      # Callee TU: extract a metadata sidecar, then assemble.
      # The sidecar contains the amdgpu.max_num_* lines (~4 lines, <200 bytes).
      # grep runs on the just-written .s file (warm page cache), near-zero cost.
      set(META_FILE "${ASM_DIR}/${fname}.${SDC_GPU_ARCH}.meta")
      add_custom_command(
        OUTPUT  ${META_FILE}
        COMMAND grep "amdgpu[.]max_num" ${ASM_FILE} > ${META_FILE}
        DEPENDS ${ASM_FILE}
        COMMENT "SPLIT[meta] ${fname}"
        VERBATIM
      )
      list(APPEND _callee_meta_files ${META_FILE})

      # -- Step B2: Assemble to device ELF (callee, no patching needed) -------
      add_custom_command(
        OUTPUT  ${DEV_OBJ}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x assembler
          -target amdgcn-amd-amdhsa
          -mcpu=${SDC_GPU_ARCH}
          -c
          -o ${DEV_OBJ} ${ASM_FILE}
        DEPENDS   ${ASM_FILE}
        COMMENT   "SPLIT[dev] ${fname}"
        VERBATIM
      )
    else()
      # Kernel TU: defer device ELF creation until after metadata patching.
      list(APPEND _kernel_fnames   "${fname}")
      list(APPEND _kernel_asm_files "${ASM_FILE}")
      list(APPEND _kernel_dev_objs "${DEV_OBJ}")
    endif()

    # -- Step C: Host-only compilation ----------------------------------------
    add_custom_command(
      OUTPUT  ${HOST_OBJ}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x hip -std=c++17
        -fgpu-rdc
        --offload-host-only
        --offload-arch=${SDC_GPU_ARCH}
        -c -O3 -fPIC
        ${_inc_flags}
        ${_def_flags}
        ${_extra_defs}
        ${_fwd_compile_opts}
        -fvisibility=hidden
        -Wno-unused-function
        -Wno-format-nonliteral
        -o ${HOST_OBJ} ${src}
      DEPENDS   ${src}
      COMMENT   "SPLIT[host] ${fname}"
      VERBATIM
    )

    # -- Step D: Bundle host + device into fat object -------------------------
    add_custom_command(
      OUTPUT  ${FAT_OBJ}
      COMMAND ${BUNDLER}
        --type=o
        --targets=${SDC_BUNDLER_HOST_TARGET},${SDC_BUNDLER_DEVICE_TARGET}
        --input=${HOST_OBJ}
        --input=${DEV_OBJ}
        --output=${FAT_OBJ}
      DEPENDS   ${HOST_OBJ} ${DEV_OBJ}
      COMMENT   "SPLIT[fat]  ${fname}"
      VERBATIM
    )

    list(APPEND ALL_FAT_OBJECTS ${FAT_OBJ})
  endforeach()

  # ---------------------------------------------------------------------------
  # Deferred kernel-TU processing: scan callee metadata sidecars and patch
  # kernel assembly so descriptors declare enough VGPRs / AGPRs / SGPRs.
  #
  # The sidecar files are tiny (~200 bytes each), so scanning 122 of them
  # takes < 0.1 s.  The kernel .s file is ~1 MB, so the read/patch/write
  # takes < 0.1 s.  Total overhead: well under a second.
  # ---------------------------------------------------------------------------
  list(LENGTH _kernel_fnames _n_kernels)
  if(_n_kernels GREATER 0)
    set(_manifest "${ASM_DIR}/callee_meta_${SDC_GPU_ARCH}.txt")
    list(JOIN _callee_meta_files "\n" _manifest_content)
    file(WRITE "${_manifest}" "${_manifest_content}\n")

    math(EXPR _last_kernel "${_n_kernels} - 1")
    foreach(_i RANGE ${_last_kernel})
      list(GET _kernel_fnames   ${_i} _fname)
      list(GET _kernel_asm_files ${_i} _asm_file)
      list(GET _kernel_dev_objs ${_i} _dev_obj)

      set(_patched_asm "${ASM_DIR}/${_fname}.${SDC_GPU_ARCH}.patched.s")

      # Step B2a: Patch kernel assembly metadata using callee sidecar files.
      # Depends on ALL callee .meta files (not .o files — no disassembly).
      add_custom_command(
        OUTPUT  ${_patched_asm}
        COMMAND ${CMAKE_COMMAND} -E copy ${_asm_file} ${_patched_asm}
        COMMAND ${CMAKE_COMMAND}
          -DASM_FILE=${_patched_asm}
          -DMANIFEST=${_manifest}
          -P ${PATCH_SCRIPT}
        DEPENDS ${_asm_file} ${_callee_meta_files} ${PATCH_SCRIPT}
        COMMENT "SPLIT[patch] ${_fname}"
        VERBATIM
      )

      # Step B2b: Assemble the patched assembly to a device ELF.
      add_custom_command(
        OUTPUT  ${_dev_obj}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x assembler
          -target amdgcn-amd-amdhsa
          -mcpu=${SDC_GPU_ARCH}
          -c
          -o ${_dev_obj} ${_patched_asm}
        DEPENDS ${_patched_asm}
        COMMENT "SPLIT[dev]  ${_fname} (patched)"
        VERBATIM
      )
    endforeach()

    message(STATUS "Split device compile: ${_n_kernels} kernel TU(s) will have metadata patched")
  endif()

  # Aggregate target so the build system can build all TUs in parallel
  add_custom_target(rccl_device_objects DEPENDS ${ALL_FAT_OBJECTS})

  # Export the list of fat objects to the parent scope
  set(DEVICE_FAT_OBJECTS ${ALL_FAT_OBJECTS} PARENT_SCOPE)

  message(STATUS "Split device compile: ${_n_sources} fat objects will be produced in ${FAT_DIR}")
endfunction()
