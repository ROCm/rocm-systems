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
#   source.cpp -> LLVM bitcode (.bc) -> assembly (.s) -> device ELF (.o) ──┐
#                                                                          │ (all TUs)
#                                                                          v
#                                                                    ld.lld -r
#                                                                          │
#                                                                          v
#                                                          clang-offload-bundler --type=bc
#                                                                   (create .hipfb blob)
#                                                                          │
#   kernel.cpp ──→ host stub (--offload-host-only -fcuda-include-gpubinary) ┘
#                                     │
#                                     v
#                          combined.fat.o (host code + .hip_fatbin section)
#
# Only kernel TUs (those defining __global__ kernels) need host stubs;
# callee TUs compiled with -DNCCL_FUNC_ONLY have no host-side registration
# to emit.  All device ELFs are pre-linked into a single relocatable object
# via ld.lld -r, then bundled into a .hipfb blob via clang-offload-bundler.
# The hipfb is embedded into the host stub object via -fcuda-include-gpubinary,
# producing a single fat object with a .hip_fatbin section (SHF_ALLOC) that
# the HIP runtime loads directly — no --hip-link or -fgpu-rdc at link time.
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
#   callee bc→asm (parallel) ──→ callee .meta ──→ kernel patch ──→ kernel asm→obj ─┐
#                             └→ callee asm→obj (parallel) ────────────────────────┤
#                                                                                  v
#   kernel bc→asm (parallel) ─────────────────────────────────   ld.lld -r (combine)
#                                                                                  │
#   kernel source ──→ host stub (--offload-host-only) ────────────────────────────┤
#                                                                                  v
#                                                                  bundler → single fat.o
#
# The only serialization points are:
#   1. The kernel patch waiting for all callee .meta files (same as before).
#   2. The ld.lld -r waiting for all device ELFs (callee + patched kernel).
# Since callee bc→asm is the same work as the original bc→obj, this adds
# no new serialization -- only the near-instant patch and fast relocatable
# link steps are inserted.
#
# Note on scratch / private segment
# ----------------------------------
# The kernel descriptor also has a private_segment_fixed_size field (scratch
# memory per work-item).  The kernel sets this to 0 because it does not itself
# spill to scratch.  Callee functions DO use scratch (up to ~1200 bytes
# cumulative), but this is covered by the uses_dynamic_stack flag which causes
# the HSA runtime to allocate a default stack budget (~8 KB) per work-item.
#
# =============================================================================
# .note YAML metadata patching
# =============================================================================
#
# Problem
# -------
# In addition to the binary kernel descriptor in .rodata (patched via the
# .set amdgpu.max_num_* directives described above), the ELF contains a
# .note section (NT_AMDGPU_METADATA) with a YAML copy of kernel metadata.
# This YAML is read by profiling and diagnostic tools such as rocprof,
# omniperf, and `llvm-objdump --notes`.
#
# Unlike the binary KD, whose register fields are computed from symbolic
# expressions referencing amdgpu.max_num_{vgpr,agpr,sgpr}, the YAML fields
# are plain integer literals emitted directly by the compiler:
#
#   .amdgpu_metadata
#   ---
#   amdhsa.version:
#     - 1
#     - 2
#   amdhsa.kernels:
#     - .agpr_count:          0      # hardcoded literal
#       .name:                ncclDevKernel_Generic_2
#       .sgpr_count:          88     # hardcoded literal
#       .uses_dynamic_stack:  true
#       .vgpr_count:          62     # hardcoded literal
#       ...
#   .end_amdgpu_metadata
#
# Patching the .set directives (Step 1 in patch_kernel_metadata.cmake) fixes
# the binary KD that the GPU hardware reads, but does NOT affect these YAML
# literals.  If left unpatched, profiling tools would report the kernel's
# own register usage (e.g. 62 VGPRs) instead of the true worst-case usage
# across all indirectly-called functions (e.g. 128 VGPRs).
#
# Solution
# --------
# The patch script (patch_kernel_metadata.cmake, Step 3) scans the kernel
# assembly for YAML entries that have ".uses_dynamic_stack: true" — the
# marker for kernels that dispatch via indirect function calls.  For each
# such entry it:
#
#   1. Locates the entry boundaries using "  - .agpr_count:" markers
#      (YAML fields are alphabetically ordered, so .agpr_count is always
#      the first field in each kernel entry).
#
#   2. Extracts the current .vgpr_count, .agpr_count, and .sgpr_count
#      literal values from the YAML text.
#
#   3. Replaces each with max(current_value, callee_maximum), where the
#      callee maximums come from the same sidecar .meta files used to
#      patch the .set directives.
#
#   4. Splices the patched entry back into the assembly text.
#
# This ensures both the hardware-facing binary kernel descriptor AND the
# tooling-facing YAML metadata reflect the true register requirements of
# IFC kernels.  The patch runs in the same cmake -P invocation as the
# .set-directive and forward-reference patches, adding negligible overhead
# (< 0.1 s for the entire script).
#
# Example: a kernel declaring 62 VGPRs, 0 AGPRs, 88 SGPRs in the YAML
# with callee maximums of 128 / 64 / 102 is rewritten to:
#
#     - .agpr_count:          64     # was 0, patched to callee max
#       ...
#       .sgpr_count:          102    # was 88, patched to callee max
#       .uses_dynamic_stack:  true
#       .vgpr_count:          128    # was 62, patched to callee max
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
        OR _opt MATCHES "^-x"
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

  set(_all_dev_objs "")
  set(_callee_meta_files "")
  set(_kernel_fnames "")
  set(_kernel_asm_files "")
  set(_kernel_dev_objs "")
  set(_kernel_srcs "")

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
    # is available as text for the kernel metadata patching step.  The asm->obj
    # step (B2) that follows is just the assembler and is very fast.
    add_custom_command(
      OUTPUT  ${ASM_FILE}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x ir
        -target amdgcn-amd-amdhsa
        -mcpu=${SDC_GPU_ARCH}
        -O3 -S
        -mllvm -amdgpu-allow-lds-in-non-entry-functions
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
      list(APPEND _all_dev_objs ${DEV_OBJ})
    else()
      # Kernel TU: defer device ELF creation until after metadata patching.
      # Host stub compilation is also deferred -- it needs the hipfb blob
      # which is created from the combined device ELF after ld.lld -r.
      list(APPEND _kernel_fnames   "${fname}")
      list(APPEND _kernel_asm_files "${ASM_FILE}")
      list(APPEND _kernel_dev_objs "${DEV_OBJ}")
      list(APPEND _kernel_srcs "${src}")
    endif()
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

  # ---------------------------------------------------------------------------
  # Combine all device ELFs into a single relocatable object via ld.lld -r,
  # then produce the final code object via ld.lld -shared.
  #
  # The relocatable link resolves internal cross-TU references and merges
  # ELF sections.  The -shared step produces an ET_DYN code object that
  # the HIP runtime can load (ET_REL is not loadable).
  # ---------------------------------------------------------------------------
  list(APPEND _all_dev_objs ${_kernel_dev_objs})

  set(LLD "${SDC_ROCM_PATH}/llvm/bin/ld.lld")
  set(COMBINED_DEV_OBJ "${DEV_DIR}/combined.${SDC_GPU_ARCH}.o")
  set(COMBINED_DEV_SO  "${DEV_DIR}/combined.${SDC_GPU_ARCH}.so")

  list(LENGTH _all_dev_objs _n_dev_objs)
  add_custom_command(
    OUTPUT  ${COMBINED_DEV_OBJ}
    COMMAND ${LLD} -r -o ${COMBINED_DEV_OBJ} ${_all_dev_objs}
    DEPENDS ${_all_dev_objs}
    COMMENT "SPLIT[link] combining ${_n_dev_objs} device objects"
    VERBATIM
  )

  add_custom_command(
    OUTPUT  ${COMBINED_DEV_SO}
    COMMAND ${LLD} -shared -o ${COMBINED_DEV_SO} ${COMBINED_DEV_OBJ}
    DEPENDS ${COMBINED_DEV_OBJ}
    COMMENT "SPLIT[cobj] producing code object"
    VERBATIM
  )

  # ---------------------------------------------------------------------------
  # Create a .hipfb blob from the code object, then compile the kernel TU
  # host stub(s) with the blob embedded as a .hip_fatbin section.
  #
  # The bundler's --type=bc creates the binary blob format (with the
  # __CLANG_OFFLOAD_BUNDLE__ magic header) rather than ELF sections.
  # The host stub is then compiled with -fcuda-include-gpubinary which
  # embeds this blob as a .hip_fatbin section (SHF_ALLOC).  The result
  # is a host object that the HIP runtime can load directly -- no
  # clang-linker-wrapper / --hip-link needed at the final link step.
  # ---------------------------------------------------------------------------
  set(COMBINED_HIPFB "${FAT_DIR}/combined.hipfb")
  add_custom_command(
    OUTPUT  ${COMBINED_HIPFB}
    COMMAND ${BUNDLER}
      --type=bc
      --targets=${SDC_BUNDLER_HOST_TARGET},${SDC_BUNDLER_DEVICE_TARGET}
      --input=/dev/null
      --input=${COMBINED_DEV_SO}
      --output=${COMBINED_HIPFB}
    DEPENDS ${COMBINED_DEV_SO}
    COMMENT "SPLIT[hipfb] creating fat binary blob"
    VERBATIM
  )

  list(LENGTH _kernel_srcs _n_kernel_srcs)
  if(_n_kernel_srcs EQUAL 0)
    message(FATAL_ERROR "Split device compile: no kernel TUs found -- need at least one for host stubs")
  endif()

  # Compile each kernel TU's host stub with the hipfb embedded.
  # --offload-host-only skips device compilation; -fcuda-include-gpubinary
  # embeds the pre-built hipfb as a .hip_fatbin section.  No -fgpu-rdc:
  # the non-RDC host registration (__hipRegisterFatBinary + __hipRegisterFunction)
  # is exactly what we need for pre-linked device code.
  set(_kernel_host_objs "")
  math(EXPR _last_ksrc "${_n_kernel_srcs} - 1")
  foreach(_i RANGE ${_last_ksrc})
    list(GET _kernel_fnames ${_i} _fname)
    list(GET _kernel_srcs   ${_i} _src)
    set(_host_obj "${HOST_DIR}/${_fname}.host.o")
    add_custom_command(
      OUTPUT  ${_host_obj}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x hip -std=c++17
        --offload-host-only
        --offload-arch=${SDC_GPU_ARCH}
        -Xclang -fcuda-include-gpubinary
        -Xclang ${COMBINED_HIPFB}
        -c -O3 -fPIC
        ${_inc_flags}
        ${_def_flags}
        ${_fwd_compile_opts}
        -fvisibility=hidden
        -Wno-unused-function
        -Wno-format-nonliteral
        -o ${_host_obj} ${_src}
      DEPENDS ${_src} ${COMBINED_HIPFB}
      COMMENT "SPLIT[host] ${_fname} (with embedded hipfb)"
      VERBATIM
    )
    list(APPEND _kernel_host_objs ${_host_obj})
  endforeach()

  if(_n_kernel_srcs EQUAL 1)
    list(GET _kernel_host_objs 0 COMBINED_FAT_OBJ)
  else()
    set(COMBINED_FAT_OBJ "${FAT_DIR}/combined.fat.o")
    add_custom_command(
      OUTPUT  ${COMBINED_FAT_OBJ}
      COMMAND ld -r -o ${COMBINED_FAT_OBJ} ${_kernel_host_objs}
      DEPENDS ${_kernel_host_objs}
      COMMENT "SPLIT[host-link] combining ${_n_kernel_srcs} kernel host stubs"
      VERBATIM
    )
  endif()

  # Aggregate target so the build system can build everything in parallel
  add_custom_target(rccl_device_objects DEPENDS ${COMBINED_FAT_OBJ})

  # Export the single fat object to the parent scope
  set(DEVICE_FAT_OBJECTS ${COMBINED_FAT_OBJ} PARENT_SCOPE)

  message(STATUS "Split device compile: ${_n_sources} device TUs -> 1 combined fat object in ${FAT_DIR}")
endfunction()
