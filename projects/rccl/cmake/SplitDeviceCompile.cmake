# cmake/SplitDeviceCompile.cmake
#
# Split device compilation pipeline for RCCL.
#
# Pre-compiles each device TU through the full LLVM backend independently and
# in parallel, links all device objects with lld into a single code object per
# GPU architecture, and bundles them with a host stub into a fat object.
#
#   source.cpp ──→ LLVM bitcode (.bc) ──→ device ELF (.o)      ← parallel
#                                                                 per arch
#
#   all device .o (per arch) ──→ ld.lld -shared ──→ combined.<arch>.so
#
#   all combined.<arch>.so ──→ clang-offload-bundler ──→ combined.hipfb
#
#   kernel TU (--offload-host-only + embedded hipfb) ──→ combined.fat.o
#
# All bc and obj compilations across all arches are independent custom
# commands, so Ninja (or Make -jN) can schedule them fully in parallel.

function(setup_split_device_compile)
  cmake_parse_arguments(SDC "COMPRESS" "TARGET;ROCM_PATH;OUTPUT_DIR;BUNDLER_HOST_TARGET" "GPU_ARCHES;SOURCES;FUNC_ONLY_SOURCES;INCLUDE_DIRS;COMPILE_DEFS;COMPILE_OPTS" ${ARGN})

  if(NOT SDC_TARGET)
    message(FATAL_ERROR "setup_split_device_compile: TARGET is required")
  endif()
  if(NOT SDC_GPU_ARCHES)
    message(FATAL_ERROR "setup_split_device_compile: GPU_ARCHES is required")
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
  list(LENGTH SDC_GPU_ARCHES _n_arches)
  message(STATUS "Split device compile: ${_n_sources} TUs × ${_n_arches} arch(es): ${SDC_GPU_ARCHES}")

  # Identify the kernel TU (there should be exactly one non-FUNC_ONLY source).
  set(_kernel_src "")
  set(_kernel_fname "")

  # ── Phase 1+2: Per-arch bc → obj, all independent custom commands ───────────
  # Every bc and obj command across all arches is a separate add_custom_command,
  # so Ninja sees them as independent nodes and can schedule them in parallel.
  set(_all_combined_dev_sos "")

  foreach(_arch ${SDC_GPU_ARCHES})
    set(_arch_dev_objects "")

    foreach(src ${SDC_SOURCES})
      get_filename_component(fname ${src} NAME_WE)

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
        COMMENT   "SPLIT[bc/${_arch}]  ${fname}"
        VERBATIM
      )

      if(_is_kernel_tu)
        # -- Step B (kernel TU): bc → asm → patch → obj ----------------------
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
          COMMENT   "SPLIT[asm+patch/${_arch}] ${fname}"
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
          COMMENT   "SPLIT[dev/${_arch}] ${fname} (from patched asm)"
          VERBATIM
        )
        # Remember the kernel TU (same source for all arches)
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
          COMMENT   "SPLIT[dev/${_arch}] ${fname}"
          VERBATIM
        )
      endif()

      list(APPEND _arch_dev_objects ${DEV_OBJ})
    endforeach()

    # -- Phase 2: Link all device objects for this arch -------------------------
    set(COMBINED_DEV_SO "${DEV_DIR}/combined.${_arch}.so")
    set(LINK_RSP        "${DEV_DIR}/combined.${_arch}.rsp")

    string(REPLACE ";" "\n" _dev_objs_rsp "${_arch_dev_objects}")
    file(WRITE "${LINK_RSP}" "${_dev_objs_rsp}\n")

    add_custom_command(
      OUTPUT  ${COMBINED_DEV_SO}
      COMMAND ${LLD} -shared -o ${COMBINED_DEV_SO} @${LINK_RSP}
      DEPENDS ${_arch_dev_objects}
      COMMENT "SPLIT[link/${_arch}] linking device objects into code object"
      VERBATIM
    )

    list(APPEND _all_combined_dev_sos ${COMBINED_DEV_SO})
  endforeach()

  # -- Phase 3: Bundle all arch .so files into a single hipfb ------------------
  set(COMBINED_HIPFB "${FAT_DIR}/combined.hipfb")

  # Build --targets and --input lists for the bundler
  set(_bundler_targets "${SDC_BUNDLER_HOST_TARGET}")
  set(_bundler_inputs  "--input=/dev/null")
  foreach(_arch ${SDC_GPU_ARCHES})
    string(APPEND _bundler_targets ",hip-amdgcn-amd-amdhsa--${_arch}")
    list(APPEND _bundler_inputs "--input=${DEV_DIR}/combined.${_arch}.so")
  endforeach()

  set(_compress_flag "")
  if(SDC_COMPRESS)
    set(_compress_flag "--compress")
  endif()

  add_custom_command(
    OUTPUT  ${COMBINED_HIPFB}
    COMMAND ${BUNDLER}
      --type=bc
      --targets=${_bundler_targets}
      ${_bundler_inputs}
      --output=${COMBINED_HIPFB}
      ${_compress_flag}
    DEPENDS ${_all_combined_dev_sos}
    COMMENT "SPLIT[hipfb] creating fat binary blob for ${SDC_GPU_ARCHES}"
    VERBATIM
  )

  # -- Phase 4: Host stub with embedded hipfb ----------------------------------
  set(COMBINED_FAT_OBJ "${FAT_DIR}/combined.fat.o")
  if(_kernel_src)
    # Build --offload-arch flags for every target arch
    set(_offload_arch_flags "")
    foreach(_arch ${SDC_GPU_ARCHES})
      list(APPEND _offload_arch_flags "--offload-arch=${_arch}")
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
        ${_fwd_compile_opts}
        -fvisibility=hidden
        -Wno-unused-function
        -Wno-format-nonliteral
        -o ${COMBINED_FAT_OBJ} ${_kernel_src}
      DEPENDS ${_kernel_src} ${COMBINED_HIPFB}
      COMMENT "SPLIT[host] ${_kernel_fname} (with embedded hipfb for ${SDC_GPU_ARCHES})"
      VERBATIM
    )
  endif()

  # Aggregate target
  add_custom_target(rccl_device_objects DEPENDS ${COMBINED_FAT_OBJ})

  # Export the single fat object to the parent scope
  set(DEVICE_FAT_OBJECTS ${COMBINED_FAT_OBJ} PARENT_SCOPE)

  math(EXPR _total_compiles "${_n_sources} * ${_n_arches}")
  message(STATUS "Split device compile: ${_total_compiles} parallel compiles (${_n_sources} TUs × ${_n_arches} arch(es)) -> 1 fat object")
endfunction()
