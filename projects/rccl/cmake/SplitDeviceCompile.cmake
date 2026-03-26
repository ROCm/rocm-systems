# cmake/SplitDeviceCompile.cmake
#
# Split device compilation pipeline for RCCL.
#
# Pre-compiles each device TU through the full LLVM backend independently and
# in parallel, links all device objects with lld into a single code object per
# architecture, and bundles them with a host stub into fat objects.
#
# For each arch in GPU_ARCHS:
#   source.cpp ──→ LLVM bitcode (.bc) ──→ device ELF (.o)      ← parallel
#
#   all device .o ──→ ld.lld -shared ──→ combined.<arch>.so     ← per-arch
#
# Then a single multi-arch bundler step:
#   combined.*.so ──→ clang-offload-bundler ──→ combined.hipfb
#
# Then per-arch host stubs:
#   kernel TU (--offload-host-only + embedded hipfb) ──→ combined.fat.<arch>.o
#
# All fat objects are linked into the final shared library without
# -fgpu-rdc / --hip-link, since device linking is already done by lld and
# the hipfb is embedded directly in the host stubs.
#
# The kernel TU (common.cu) is compiled to assembly (.s) so we can patch
# the .set directives for the Generic kernel descriptors to provision the
# known callee maximums (128 VGPRs, 64 AGPRs, flat scratch, dynamic stack)
# before assembling.

function(setup_split_device_compile)
  cmake_parse_arguments(SDC "" "TARGET;ROCM_PATH;OUTPUT_DIR;BUNDLER_HOST_TARGET" "GPU_ARCHS;SOURCES;FUNC_ONLY_SOURCES;INCLUDE_DIRS;COMPILE_DEFS;COMPILE_OPTS;BUNDLER_DEVICE_TARGETS" ${ARGN})

  if(NOT SDC_TARGET)
    message(FATAL_ERROR "setup_split_device_compile: TARGET is required")
  endif()
  if(NOT SDC_GPU_ARCHS)
    message(FATAL_ERROR "setup_split_device_compile: GPU_ARCHS is required")
  endif()
  if(NOT SDC_ROCM_PATH)
    set(SDC_ROCM_PATH "")
  endif()
  if(NOT SDC_OUTPUT_DIR)
    message(FATAL_ERROR "setup_split_device_compile: OUTPUT_DIR is required")
  endif()
  if(NOT SDC_BUNDLER_HOST_TARGET)
    message(FATAL_ERROR "setup_split_device_compile: BUNDLER_HOST_TARGET is required")
  endif()

  # Derive LLVM tool paths from the compiler location so that
  # both /opt/rocm installs and TheRock/CI build trees work.
  get_filename_component(_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
  find_program(BUNDLER clang-offload-bundler
    HINTS "${_compiler_dir}" "${SDC_ROCM_PATH}/llvm/bin" NO_DEFAULT_PATH)
  if(NOT BUNDLER)
    find_program(BUNDLER clang-offload-bundler)
  endif()
  find_program(LLD ld.lld
    HINTS "${_compiler_dir}" "${SDC_ROCM_PATH}/llvm/bin" NO_DEFAULT_PATH)
  if(NOT LLD)
    find_program(LLD ld.lld)
  endif()
  if(NOT BUNDLER)
    message(FATAL_ERROR "Split device compile: clang-offload-bundler not found")
  endif()
  if(NOT LLD)
    message(FATAL_ERROR "Split device compile: ld.lld not found")
  endif()

  # Output directories
  set(BC_DIR   "${SDC_OUTPUT_DIR}/split_device/bc")
  set(DEV_DIR  "${SDC_OUTPUT_DIR}/split_device/dev_obj")
  set(HOST_DIR "${SDC_OUTPUT_DIR}/split_device/host_obj")
  set(FAT_DIR  "${SDC_OUTPUT_DIR}/split_device/fat_obj")
  file(MAKE_DIRECTORY ${BC_DIR} ${DEV_DIR} ${HOST_DIR} ${FAT_DIR})

  # Build the include-directory flags list
  set(_inc_flags "")
  foreach(_dir ${SDC_INCLUDE_DIRS})
    list(APPEND _inc_flags "-I${_dir}")
  endforeach()

  # Build the compile-definition flags list
  set(_def_flags "")
  foreach(_def ${SDC_COMPILE_DEFS})
    list(APPEND _def_flags "-D${_def}")
  endforeach()

  # Forward compile options, filtering out flags managed by this pipeline
  set(_fwd_compile_opts "")
  set(_skip_next OFF)
  foreach(_opt ${SDC_COMPILE_OPTS})
    if(_skip_next)
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
        OR _opt MATCHES "^-fPIC"
        OR _opt MATCHES "^-fsanitize="
        OR _opt MATCHES "^-shared-libasan")
      # Skip: already set explicitly or irrelevant for device code
    elseif(_opt STREQUAL "-mllvm")
      list(APPEND _fwd_compile_opts "${_opt}")
      set(_skip_next ON)
    else()
      list(APPEND _fwd_compile_opts "${_opt}")
    endif()
  endforeach()

  list(LENGTH SDC_SOURCES _n_sources)
  list(LENGTH SDC_GPU_ARCHS _n_archs)
  message(STATUS "Split device compile: ${_n_sources} TUs × ${_n_archs} architectures")

  set(_kernel_src "")
  set(_kernel_fname "")
  set(ALL_COMBINED_DEV_SOS "")
  set(ALL_FAT_OBJECTS "")

  # ── Per-architecture loop ────────────────────────────────────────────────────
  foreach(_arch ${SDC_GPU_ARCHS})
    set(_arch_dev_objects "")

    foreach(src ${SDC_SOURCES})
      get_filename_component(fname ${src} NAME_WE)

      # Determine if this source needs -DRCCL_SPLIT_DEVICE_TU
      set(_extra_defs "")
      set(_is_kernel_tu FALSE)
      list(FIND SDC_FUNC_ONLY_SOURCES "${src}" _func_only_idx)
      if(NOT _func_only_idx EQUAL -1)
        set(_extra_defs "-DRCCL_SPLIT_DEVICE_TU")
      else()
        set(_is_kernel_tu TRUE)
        if(_arch STREQUAL "gfx950")
          set(_extra_defs "-DRCCL_ARGS_IN_SCRATCH")
        endif()
      endif()

      set(BC_FILE  "${BC_DIR}/${fname}.${_arch}.bc")
      set(DEV_OBJ  "${DEV_DIR}/${fname}.${_arch}.o")

      # -- Step A: Compile to LLVM bitcode (device only) ----------------------
      add_custom_command(
        OUTPUT  ${BC_FILE}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x hip -std=c++17
          -fgpu-rdc
          --offload-device-only
          --offload-arch=${_arch}
          -emit-llvm -c -O3 -gline-tables-only
          ${_inc_flags}
          ${_def_flags}
          ${_extra_defs}
          ${_fwd_compile_opts}
          -fvisibility=hidden
          -Wno-unused-function
          -Wno-format-nonliteral
          -o ${BC_FILE} ${src}
        DEPENDS   ${src}
        COMMENT   "SPLIT[bc]  ${fname} [${_arch}]"
        VERBATIM
      )

      if(_is_kernel_tu)
        # -- Step B (kernel TU): bc → asm → patch kernel descriptors → obj ----
        set(ASM_FILE "${DEV_DIR}/${fname}.${_arch}.s")

        set(_sed_args
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.num_vgpr,\\).*/\\1 128/"
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.num_agpr,\\).*/\\1 64/"
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.numbered_sgpr,\\).*/\\1 102/"
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.uses_flat_scratch,\\).*/\\1 1/"
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.has_dyn_sized_stack,\\).*/\\1 1/"
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.has_indirect_call,\\).*/\\1 1/"
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.has_recursion,\\).*/\\1 1/"
          -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*\\.num_named_barrier,\\).*/\\1 0/"
          -e "s/\\.amdhsa_next_free_vgpr .*/\\.amdhsa_next_free_vgpr 192/"
          -e "s/\\.amdhsa_accum_offset .*/\\.amdhsa_accum_offset 128/"
          -e "s/\\.amdhsa_next_free_sgpr .*/\\.amdhsa_next_free_sgpr 102/"
          -e "s/^\\([[:space:]]*\\.vgpr_count:\\).*/\\1     192/"
          -e "s/^\\([[:space:]]*\\.sgpr_count:\\).*/\\1     102/"
          -e "s/^\\([[:space:]]*\\)\\(- \\)\\?\\(\\.agpr_count:\\).*/\\1\\2\\3     64/"
          -e "s/^\\([[:space:]]*\\.uses_dynamic_stack:\\).*/\\1 true/"
          -e "s/\\(.section[[:space:]]\\+asan_globals,[^,]*,@progbits,[^,]*\\),.*/\\1,comdat/"
        )

        add_custom_command(
          OUTPUT  ${ASM_FILE}
          COMMAND ${CMAKE_CXX_COMPILER}
            -x ir
            -target amdgcn-amd-amdhsa
            -mcpu=${_arch}
            -O3 -S -gline-tables-only
            -o ${ASM_FILE} ${BC_FILE}
          COMMAND sed -i ${_sed_args} ${ASM_FILE}
          DEPENDS   ${BC_FILE}
          COMMENT   "SPLIT[asm+patch] ${fname} [${_arch}]"
          VERBATIM
        )
        add_custom_command(
          OUTPUT  ${DEV_OBJ}
          COMMAND ${CMAKE_CXX_COMPILER}
            -x assembler
            -target amdgcn-amd-amdhsa
            -mcpu=${_arch}
            -c -gline-tables-only
            -o ${DEV_OBJ} ${ASM_FILE}
          DEPENDS   ${ASM_FILE}
          COMMENT   "SPLIT[dev] ${fname} [${_arch}] (from patched asm)"
          VERBATIM
        )
        # Remember the kernel source (same for all arches)
        set(_kernel_src   ${src})
        set(_kernel_fname ${fname})
      else()
        # -- Step B (non-kernel TU): bc → obj directly ------------------------
        add_custom_command(
          OUTPUT  ${DEV_OBJ}
          COMMAND ${CMAKE_CXX_COMPILER}
            -x ir
            -target amdgcn-amd-amdhsa
            -mcpu=${_arch}
            -O3 -c -gline-tables-only
            -o ${DEV_OBJ} ${BC_FILE}
          DEPENDS   ${BC_FILE}
          COMMENT   "SPLIT[dev] ${fname} [${_arch}]"
          VERBATIM
        )
      endif()

      list(APPEND _arch_dev_objects ${DEV_OBJ})
    endforeach()

    # -- Link all device objects for this arch with lld -----------------------
    set(COMBINED_DEV_SO "${DEV_DIR}/combined.${_arch}.so")
    set(LINK_RSP        "${DEV_DIR}/combined.${_arch}.rsp")

    string(REPLACE ";" "\n" _dev_objs_rsp "${_arch_dev_objects}")
    file(WRITE "${LINK_RSP}" "${_dev_objs_rsp}\n")

    list(LENGTH _arch_dev_objects _n_arch_objs)
    add_custom_command(
      OUTPUT  ${COMBINED_DEV_SO}
      COMMAND ${LLD} -shared --compress-debug-sections=zlib -o ${COMBINED_DEV_SO} @${LINK_RSP}
      DEPENDS ${_arch_dev_objects}
      COMMENT "SPLIT[link] linking ${_n_arch_objs} device objects for ${_arch}"
      VERBATIM
    )

    list(APPEND ALL_COMBINED_DEV_SOS ${COMBINED_DEV_SO})
    message(STATUS "  ${_arch}: ${_n_arch_objs} TUs")
  endforeach()

  # ── Single multi-arch bundler step ─────────────────────────────────────────
  set(COMBINED_HIPFB "${FAT_DIR}/combined.hipfb")

  # Build --targets as comma-separated, --input as separate flags per file:
  #   --targets=host-x86_64-...,hip-amdgcn-...--gfx908,hip-amdgcn-...--gfx90a,...
  #   --input=/dev/null --input=combined.gfx908.so --input=combined.gfx90a.so ...
  set(_bundler_targets "${SDC_BUNDLER_HOST_TARGET}")
  set(_bundler_input_flags --input=/dev/null)
  foreach(_idx RANGE 0 ${_n_archs})
    if(_idx LESS _n_archs)
      list(GET SDC_BUNDLER_DEVICE_TARGETS ${_idx} _dev_target)
      set(_bundler_targets "${_bundler_targets},${_dev_target}")
      list(GET ALL_COMBINED_DEV_SOS ${_idx} _so)
      list(APPEND _bundler_input_flags --input=${_so})
    endif()
  endforeach()

  add_custom_command(
    OUTPUT  ${COMBINED_HIPFB}
    COMMAND ${BUNDLER}
      --type=bc
      --targets=${_bundler_targets}
      ${_bundler_input_flags}
      --output=${COMBINED_HIPFB}
    DEPENDS ${ALL_COMBINED_DEV_SOS}
    COMMENT "SPLIT[hipfb] bundling ${_n_archs} architectures into fat binary"
    VERBATIM
  )

  # ── Host stub with embedded hipfb (single compilation, all arches) ────────
  set(COMBINED_FAT_OBJ "${FAT_DIR}/combined.fat.o")
  if(_kernel_src)
    set(_offload_arch_flags "")
    set(_host_extra_defs "")
    foreach(_arch ${SDC_GPU_ARCHS})
      list(APPEND _offload_arch_flags --offload-arch=${_arch})
      if(_arch STREQUAL "gfx950")
        set(_host_extra_defs "-DRCCL_ARGS_IN_SCRATCH")
      endif()
    endforeach()

    add_custom_command(
      OUTPUT  ${COMBINED_FAT_OBJ}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x hip -std=c++17
        --offload-host-only
        ${_offload_arch_flags}
        -Xclang -fcuda-include-gpubinary
        -Xclang ${COMBINED_HIPFB}
        -c -O3 -fPIC
        ${_inc_flags}
        ${_def_flags}
        ${_host_extra_defs}
        ${_fwd_compile_opts}
        -fvisibility=hidden
        -Wno-unused-function
        -Wno-format-nonliteral
        -o ${COMBINED_FAT_OBJ} ${_kernel_src}
      DEPENDS ${_kernel_src} ${COMBINED_HIPFB}
      COMMENT "SPLIT[host] ${_kernel_fname} (with embedded hipfb for ${_n_archs} archs)"
      VERBATIM
    )
    list(APPEND ALL_FAT_OBJECTS ${COMBINED_FAT_OBJ})
  endif()

  # Aggregate target
  add_custom_target(rccl_device_objects DEPENDS ${ALL_FAT_OBJECTS})

  # Export all fat objects to the parent scope
  set(DEVICE_FAT_OBJECTS ${ALL_FAT_OBJECTS} PARENT_SCOPE)

  list(LENGTH ALL_FAT_OBJECTS _n_fat)
  message(STATUS "Split device compile: ${_n_sources} TUs × ${_n_archs} archs -> ${_n_fat} fat object(s) in ${FAT_DIR}")
endfunction()
