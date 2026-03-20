# cmake/SplitDeviceCompile.cmake
#
# Split device compilation pipeline for RCCL.
#
# Pre-compiles each device TU through the full LLVM backend independently and
# in parallel, links all device objects with lld into a single code object,
# and bundles it with a host stub into a fat object.
#
#   source.cpp ──→ LLVM bitcode (.bc) ──→ device ELF (.o)      ← parallel
#
#   all device .o ──→ ld.lld -shared ──→ combined.<arch>.so
#                 ──→ clang-offload-bundler --type=bc ──→ combined.hipfb
#
#   kernel TU (--offload-host-only + embedded hipfb) ──→ combined.fat.o
#
# The single fat object is then linked into the final shared library without
# -fgpu-rdc / --hip-link, since device linking is already done by lld and
# the hipfb is embedded directly in the host stub.
#
# The kernel TU (common.cu) is compiled to assembly (.s) so we can patch
# the .set directives for the Generic kernel descriptors to provision the
# known callee maximums (128 VGPRs, 64 AGPRs, flat scratch, dynamic stack)
# before assembling.  This replaces the Python patch_kernel_descriptor.py
# script and avoids scanning all device objects at build time.
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
    set(SDC_ROCM_PATH "")
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

  # Derive LLVM tool paths from the compiler location so that
  # both /opt/rocm installs and TheRock/CI build trees.
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

  set(ALL_DEV_OBJECTS "")
  set(_kernel_src "")
  set(_kernel_fname "")

  list(LENGTH SDC_SOURCES _n_sources)
  message(STATUS "Split device compile: ${_n_sources} TUs for ${SDC_GPU_ARCH}")

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
      if(SDC_GPU_ARCH STREQUAL "gfx950")
        set(_extra_defs "-DRCCL_ARGS_IN_SCRATCH")
      endif()
    endif()

    set(BC_FILE  "${BC_DIR}/${fname}.${SDC_GPU_ARCH}.bc")
    set(DEV_OBJ  "${DEV_DIR}/${fname}.${SDC_GPU_ARCH}.o")

    # -- Step A: Compile to LLVM bitcode (device only) ------------------------
    set(_rdc_flag "-fgpu-rdc")
    add_custom_command(
      OUTPUT  ${BC_FILE}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x hip -std=c++17
        ${_rdc_flag}
        --offload-device-only
        --offload-arch=${SDC_GPU_ARCH}
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
      COMMENT   "SPLIT[bc]  ${fname}"
      VERBATIM
    )

    if(_is_kernel_tu)
      # -- Step B (kernel TU): bc → asm → patch kernel descriptors → obj ------
      # The kernel dispatches device functions through a function pointer table.
      # The compiler cannot propagate callee register requirements through the
      # indirection, so we patch the .set directives in the assembly to provision
      # the known maximums: 128 VGPRs, 64 AGPRs, flat scratch, dynamic stack.
      set(ASM_FILE "${DEV_DIR}/${fname}.${SDC_GPU_ARCH}.s")

      set(_sed_args
        # -- Patch .set directives (kernel descriptor binary) --
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.num_vgpr,\\).*/\\1 128/"
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.num_agpr,\\).*/\\1 64/"
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.numbered_sgpr,\\).*/\\1 102/"
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.uses_flat_scratch,\\).*/\\1 1/"
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.has_dyn_sized_stack,\\).*/\\1 1/"
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.has_indirect_call,\\).*/\\1 1/"
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*ncclDevKernel[^.]*\\.has_recursion,\\).*/\\1 1/"
        -e "s/^\\([[:space:]]*\\.set[[:space:]]\\+.*\\.num_named_barrier,\\).*/\\1 0/"
        # -- Patch .amdhsa_kernel directives --
        -e "s/\\.amdhsa_next_free_vgpr .*/\\.amdhsa_next_free_vgpr 192/"
        -e "s/\\.amdhsa_accum_offset .*/\\.amdhsa_accum_offset 128/"
        -e "s/\\.amdhsa_next_free_sgpr .*/\\.amdhsa_next_free_sgpr 102/"
        # -- Patch .amdgpu_metadata YAML (must match the KD values) --
        -e "s/^\\([[:space:]]*\\.vgpr_count:\\).*/\\1     192/"
        -e "s/^\\([[:space:]]*\\.sgpr_count:\\).*/\\1     102/"
        -e "s/^\\([[:space:]]*\\)\\(- \\)\\?\\(\\.agpr_count:\\).*/\\1\\2\\3     64/"
        -e "s/^\\([[:space:]]*\\.uses_dynamic_stack:\\).*/\\1 true/"
        # -- ASAN: rewrite malformed .section asan_globals directives --
        # The AMDGCN backend emits a COMDAT signature + ,unique,N suffix
        # that the assembler rejects.  Replace everything after the symbol
        # name (4th comma-field) with just "comdat".
        -e "s/\\(.section[[:space:]]\\+asan_globals,[^,]*,@progbits,[^,]*\\),.*/\\1,comdat/"
      )

      add_custom_command(
        OUTPUT  ${ASM_FILE}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x ir
          -target amdgcn-amd-amdhsa
          -mcpu=${SDC_GPU_ARCH}
          -O3 -S -gline-tables-only
          -o ${ASM_FILE} ${BC_FILE}
        COMMAND sed -i ${_sed_args} ${ASM_FILE}
        DEPENDS   ${BC_FILE}
        COMMENT   "SPLIT[asm+patch] ${fname}"
        VERBATIM
      )
      add_custom_command(
        OUTPUT  ${DEV_OBJ}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x assembler
          -target amdgcn-amd-amdhsa
          -mcpu=${SDC_GPU_ARCH}
          -c -gline-tables-only
          -o ${DEV_OBJ} ${ASM_FILE}
        DEPENDS   ${ASM_FILE}
        COMMENT   "SPLIT[dev] ${fname} (from patched asm)"
        VERBATIM
      )
      set(_kernel_src   ${src})
      set(_kernel_fname ${fname})
    else()
      # -- Step B (non-kernel TU): bc → obj directly --------------------------
      add_custom_command(
        OUTPUT  ${DEV_OBJ}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x ir
          -target amdgcn-amd-amdhsa
          -mcpu=${SDC_GPU_ARCH}
          -O3 -c -gline-tables-only
          -o ${DEV_OBJ} ${BC_FILE}
        DEPENDS   ${BC_FILE}
        COMMENT   "SPLIT[dev] ${fname}"
        VERBATIM
      )
    endif()

    list(APPEND ALL_DEV_OBJECTS ${DEV_OBJ})
  endforeach()

  # -- Link all device objects with lld ---------------------------------------
  set(COMBINED_DEV_SO "${DEV_DIR}/combined.${SDC_GPU_ARCH}.so")
  set(LINK_RSP        "${DEV_DIR}/combined.${SDC_GPU_ARCH}.rsp")

  string(REPLACE ";" "\n" _dev_objs_rsp "${ALL_DEV_OBJECTS}")
  file(WRITE "${LINK_RSP}" "${_dev_objs_rsp}\n")

  add_custom_command(
    OUTPUT  ${COMBINED_DEV_SO}
    COMMAND ${LLD} -shared -o ${COMBINED_DEV_SO} @${LINK_RSP}
    DEPENDS ${ALL_DEV_OBJECTS}
    COMMENT "SPLIT[link] linking ${_n_sources} device objects into code object for ${SDC_GPU_ARCH}"
    VERBATIM
  )

  # -- Bundle device .so into hipfb ------------------------------------------
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
    COMMENT "SPLIT[hipfb] creating fat binary blob for ${SDC_GPU_ARCH}"
    VERBATIM
  )

  # -- Host stub with embedded hipfb -----------------------------------------
  # Compiles the kernel TU host-only and embeds the hipfb so the HIP runtime
  # can find and load the device code at launch time.
  set(COMBINED_FAT_OBJ "${FAT_DIR}/combined.fat.o")
  if(_kernel_src)
    add_custom_command(
      OUTPUT  ${COMBINED_FAT_OBJ}
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
        -o ${COMBINED_FAT_OBJ} ${_kernel_src}
      DEPENDS ${_kernel_src} ${COMBINED_HIPFB}
      COMMENT "SPLIT[host] ${_kernel_fname} (with embedded hipfb)"
      VERBATIM
    )
  endif()

  # Aggregate target
  add_custom_target(rccl_device_objects DEPENDS ${COMBINED_FAT_OBJ})

  # Export the single fat object to the parent scope
  set(DEVICE_FAT_OBJECTS ${COMBINED_FAT_OBJ} PARENT_SCOPE)

  message(STATUS "Split device compile: ${_n_sources} TUs -> 1 fat object in ${FAT_DIR}")
endfunction()
