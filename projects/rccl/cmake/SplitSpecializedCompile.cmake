# cmake/SplitSpecializedCompile.cmake
#
# Split specialized kernel compilation pipeline for RCCL.
#
# Replaces the custom device_linker with standard LLVM tools:
#   - Specialized kernels: compile with wrapping kernel for correct LDS/codegen,
#     then strip the kernel from assembly, keeping only ncclDevFunc_*.
#   - Dispatcher (common.cu): compile, patch register metadata from callee
#     sidecars, assemble.
#   - Link all device objects with lld, bundle into hipfb, embed in host stubs.
#
# onerank.cu is compiled via the normal HIP build (self-contained, no cross-TU
# linking needed).  Keeping it separate avoids -fgpu-rdc symbol mangling of its
# anonymous-namespace kernels.
#
# Pipeline per source category:
#
#   Specialized (.cpp from generate_specialized.py):
#     source → bc (-fgpu-rdc, -DSPECIALIZED_KERNEL=1)
#            → asm → strip_kernel.py (remove kernel, extract .meta)
#            → assemble → dev_obj/<arch>/*.o
#
#   Dispatcher (common.cu):
#     source → bc (-fgpu-rdc, -DUSE_INDIRECT_FUNCTION_CALL, includes device_table.h)
#            → asm → patch_kernel_metadata.cmake (register maximums from .meta)
#            → assemble → dev_obj/<arch>/common.o
#            + host stub (--offload-host-only with embedded hipfb)
#
#   Link stage (per arch):
#     all dev_obj/<arch>/*.o → lld -r → combined.<arch>.o → lld -shared → combined.<arch>.so
#
#   Bundle stage:
#     all combined.<arch>.so → clang-offload-bundler → combined.hipfb
#     host stubs → ld -r → combined.fat.o
#

function(setup_split_specialized_compile)
  cmake_parse_arguments(SSC "ENABLE_COMPRESS"
    "TARGET;ROCM_PATH;OUTPUT_DIR;BUNDLER_HOST_TARGET"
    "GPU_ARCHS;BUNDLER_DEVICE_TARGETS;SPECIALIZED_SOURCES;DISPATCHER_SOURCES;PASSTHROUGH_KERNEL_SOURCES;INCLUDE_DIRS;COMPILE_DEFS;COMPILE_OPTS"
    ${ARGN})

  # Validate required args
  foreach(_arg TARGET ROCM_PATH OUTPUT_DIR BUNDLER_HOST_TARGET)
    if(NOT SSC_${_arg})
      message(FATAL_ERROR "setup_split_specialized_compile: ${_arg} is required")
    endif()
  endforeach()
  if(NOT SSC_GPU_ARCHS)
    message(FATAL_ERROR "setup_split_specialized_compile: GPU_ARCHS is required")
  endif()
  if(NOT SSC_SPECIALIZED_SOURCES)
    message(FATAL_ERROR "setup_split_specialized_compile: SPECIALIZED_SOURCES is required")
  endif()
  if(NOT SSC_DISPATCHER_SOURCES)
    message(FATAL_ERROR "setup_split_specialized_compile: DISPATCHER_SOURCES is required")
  endif()

  set(CLANG      "${SSC_ROCM_PATH}/llvm/bin/clang")
  set(BUNDLER    "${SSC_ROCM_PATH}/llvm/bin/clang-offload-bundler")
  set(LLD        "${SSC_ROCM_PATH}/llvm/bin/ld.lld")
  set(STRIP_PY   "${PROJECT_SOURCE_DIR}/tools/split_specialized/strip_kernel.py")
  set(PATCH_CMAKE "${PROJECT_SOURCE_DIR}/cmake/scripts/patch_kernel_metadata.cmake")

  # Top-level output directories
  set(BC_DIR   "${SSC_OUTPUT_DIR}/split_specialized/bc")
  set(ASM_DIR  "${SSC_OUTPUT_DIR}/split_specialized/asm")
  set(META_DIR "${SSC_OUTPUT_DIR}/split_specialized/meta")
  set(DEV_DIR  "${SSC_OUTPUT_DIR}/split_specialized/dev_obj")
  set(HOST_DIR "${SSC_OUTPUT_DIR}/split_specialized/host_obj")
  set(FAT_DIR  "${SSC_OUTPUT_DIR}/split_specialized/fat_obj")

  # Build include/definition/option flag lists
  set(_inc_flags "")
  foreach(_dir ${SSC_INCLUDE_DIRS})
    list(APPEND _inc_flags "-I${_dir}")
  endforeach()

  set(_def_flags "")
  foreach(_def ${SSC_COMPILE_DEFS})
    list(APPEND _def_flags "-D${_def}")
  endforeach()

  # Filter compile options
  set(_fwd_opts "")
  set(_skip_next OFF)
  foreach(_opt ${SSC_COMPILE_OPTS})
    if(_skip_next)
      list(APPEND _fwd_opts "${_opt}")
      set(_skip_next OFF)
    elseif(_opt MATCHES "^-parallel-jobs"
        OR _opt MATCHES "^--offload-compress"
        OR _opt MATCHES "^--offload-arch"
        OR _opt MATCHES "^-fvisibility"
        OR _opt MATCHES "^-fgpu-rdc"
        OR _opt MATCHES "^-fno-gpu-rdc"
        OR _opt MATCHES "^-x"
        OR _opt MATCHES "^-std="
        OR _opt MATCHES "^-O[0-3s]$"
        OR _opt MATCHES "^-fPIC")
      # skip
    elseif(_opt STREQUAL "-mllvm")
      list(APPEND _fwd_opts "${_opt}")
      set(_skip_next ON)
    else()
      list(APPEND _fwd_opts "${_opt}")
    endif()
  endforeach()

  list(LENGTH SSC_SPECIALIZED_SOURCES _n_spec)
  list(LENGTH SSC_GPU_ARCHS _n_archs)
  message(STATUS "Split specialized: ${_n_spec} specialized kernels × ${_n_archs} arch(es)")

  # Collect all per-arch code objects for bundling
  set(_all_combined_sos "")

  # =========================================================================
  # Per-architecture device compilation pipeline
  # =========================================================================
  foreach(_arch ${SSC_GPU_ARCHS})
    # Per-arch subdirectories
    set(_BC_DIR   "${BC_DIR}/${_arch}")
    set(_ASM_DIR  "${ASM_DIR}/${_arch}")
    set(_META_DIR "${META_DIR}/${_arch}")
    set(_DEV_DIR  "${DEV_DIR}/${_arch}")
    file(MAKE_DIRECTORY ${_BC_DIR} ${_ASM_DIR} ${_META_DIR} ${_DEV_DIR})

    set(_arch_dev_objs "")
    set(_arch_meta_files "")

    # =======================================================================
    # Specialized kernels: bc → asm → strip + meta → assemble
    # =======================================================================
    foreach(src ${SSC_SPECIALIZED_SOURCES})
      get_filename_component(fname ${src} NAME_WE)

      set(BC_FILE      "${_BC_DIR}/${fname}.bc")
      set(ASM_FILE     "${_ASM_DIR}/${fname}.s")
      set(STRIPPED_ASM  "${_ASM_DIR}/${fname}.stripped.s")
      set(META_FILE    "${_META_DIR}/${fname}.meta")
      set(DEV_OBJ      "${_DEV_DIR}/${fname}.o")

      add_custom_command(
        OUTPUT  ${BC_FILE}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x hip -std=c++17
          -fgpu-rdc
          --offload-device-only
          --offload-arch=${_arch}
          -emit-llvm -c -gline-tables-only -O3 -DNDEBUG=1
          -DSPECIALIZED_KERNEL=1
          -DUSE_INDIRECT_FUNCTION_CALL
          ${_def_flags}
          ${_inc_flags}
          ${_fwd_opts}
          -fvisibility=hidden
          -Wno-unused-function
          -Wno-format-nonliteral
          -o ${BC_FILE} ${src}
        DEPENDS   ${src}
        COMMENT   "SPLIT[bc/${_arch}]  ${fname}"
        VERBATIM
      )

      add_custom_command(
        OUTPUT  ${ASM_FILE}
        COMMAND ${CLANG}
          -x ir
          -target amdgcn-amd-amdhsa
          -mcpu=${_arch}
          -gline-tables-only -O3 -S
          -o ${ASM_FILE} ${BC_FILE}
        DEPENDS   ${BC_FILE}
        COMMENT   "SPLIT[asm/${_arch}] ${fname}"
        VERBATIM
      )

      add_custom_command(
        OUTPUT  ${STRIPPED_ASM} ${META_FILE}
        COMMAND ${Python3_EXECUTABLE} ${STRIP_PY}
          ${ASM_FILE} ${STRIPPED_ASM} --meta ${META_FILE}
        DEPENDS ${ASM_FILE} ${STRIP_PY}
        COMMENT "SPLIT[strip/${_arch}] ${fname}"
        VERBATIM
      )
      list(APPEND _arch_meta_files ${META_FILE})

      add_custom_command(
        OUTPUT  ${DEV_OBJ}
        COMMAND ${CLANG}
          -x assembler
          -target amdgcn-amd-amdhsa
          -mcpu=${_arch}
          -c
          -o ${DEV_OBJ} ${STRIPPED_ASM}
        DEPENDS   ${STRIPPED_ASM}
        COMMENT   "SPLIT[dev/${_arch}] ${fname}"
        VERBATIM
      )
      list(APPEND _arch_dev_objs ${DEV_OBJ})
    endforeach()

    # =======================================================================
    # Write the callee metadata manifest for this arch
    # =======================================================================
    set(_manifest "${_META_DIR}/manifest.txt")
    list(JOIN _arch_meta_files "\n" _manifest_content)
    file(WRITE "${_manifest}" "${_manifest_content}\n")

    # =======================================================================
    # Dispatcher kernels: bc → asm → patch → assemble
    # =======================================================================
    set(_arch_kernel_fnames "")
    set(_arch_kernel_srcs "")

    foreach(src ${SSC_DISPATCHER_SOURCES})
      get_filename_component(fname ${src} NAME_WE)

      set(BC_FILE      "${_BC_DIR}/${fname}.bc")
      set(ASM_FILE     "${_ASM_DIR}/${fname}.s")
      set(PATCHED_ASM  "${_ASM_DIR}/${fname}.patched.s")
      set(DEV_OBJ      "${_DEV_DIR}/${fname}.o")

      add_custom_command(
        OUTPUT  ${BC_FILE}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x hip -std=c++17
          -fgpu-rdc
          --offload-device-only
          --offload-arch=${_arch}
          -emit-llvm -c -g -O3
          -DUSE_INDIRECT_FUNCTION_CALL
          ${_def_flags}
          ${_inc_flags}
          ${_fwd_opts}
          -fvisibility=hidden
          -Wno-unused-function
          -Wno-format-nonliteral
          -o ${BC_FILE} ${src}
        DEPENDS   ${src}
        COMMENT   "SPLIT[bc/${_arch}]  ${fname} (dispatcher)"
        VERBATIM
      )

      add_custom_command(
        OUTPUT  ${ASM_FILE}
        COMMAND ${CLANG}
          -x ir
          -target amdgcn-amd-amdhsa
          -mcpu=${_arch}
          -g -O3 -S
          -o ${ASM_FILE} ${BC_FILE}
        DEPENDS   ${BC_FILE}
        COMMENT   "SPLIT[asm/${_arch}] ${fname} (dispatcher)"
        VERBATIM
      )

      add_custom_command(
        OUTPUT  ${PATCHED_ASM}
        COMMAND ${CMAKE_COMMAND} -E copy ${ASM_FILE} ${PATCHED_ASM}
        COMMAND ${CMAKE_COMMAND}
          -DASM_FILE=${PATCHED_ASM}
          -DMANIFEST=${_manifest}
          -P ${PATCH_CMAKE}
        DEPENDS ${ASM_FILE} ${_arch_meta_files} ${PATCH_CMAKE}
        COMMENT "SPLIT[patch/${_arch}] ${fname}"
        VERBATIM
      )

      add_custom_command(
        OUTPUT  ${DEV_OBJ}
        COMMAND ${CLANG}
          -x assembler
          -target amdgcn-amd-amdhsa
          -mcpu=${_arch}
          -c
          -o ${DEV_OBJ} ${PATCHED_ASM}
        DEPENDS   ${PATCHED_ASM}
        COMMENT   "SPLIT[dev/${_arch}] ${fname} (dispatcher, patched)"
        VERBATIM
      )
      list(APPEND _arch_dev_objs ${DEV_OBJ})
      list(APPEND _arch_kernel_fnames "${fname}")
      list(APPEND _arch_kernel_srcs "${src}")
    endforeach()

    # =======================================================================
    # Optional passthrough kernel TUs
    # =======================================================================
    foreach(src ${SSC_PASSTHROUGH_KERNEL_SOURCES})
      get_filename_component(fname ${src} NAME_WE)

      set(BC_FILE  "${_BC_DIR}/${fname}.bc")
      set(ASM_FILE "${_ASM_DIR}/${fname}.s")
      set(DEV_OBJ  "${_DEV_DIR}/${fname}.o")

      add_custom_command(
        OUTPUT  ${BC_FILE}
        COMMAND ${CMAKE_CXX_COMPILER}
          -x hip -std=c++17
          -fgpu-rdc
          --offload-device-only
          --offload-arch=${_arch}
          -emit-llvm -c -O3
          ${_def_flags}
          ${_inc_flags}
          ${_fwd_opts}
          -fvisibility=hidden
          -Wno-unused-function
          -Wno-format-nonliteral
          -o ${BC_FILE} ${src}
        DEPENDS   ${src}
        COMMENT   "SPLIT[bc/${_arch}]  ${fname} (passthrough)"
        VERBATIM
      )

      add_custom_command(
        OUTPUT  ${ASM_FILE}
        COMMAND ${CLANG}
          -x ir
          -target amdgcn-amd-amdhsa
          -mcpu=${_arch}
          -O3 -S
          -o ${ASM_FILE} ${BC_FILE}
        DEPENDS   ${BC_FILE}
        COMMENT   "SPLIT[asm/${_arch}] ${fname} (passthrough)"
        VERBATIM
      )

      add_custom_command(
        OUTPUT  ${DEV_OBJ}
        COMMAND ${CLANG}
          -x assembler
          -target amdgcn-amd-amdhsa
          -mcpu=${_arch}
          -c
          -o ${DEV_OBJ} ${ASM_FILE}
        DEPENDS   ${ASM_FILE}
        COMMENT   "SPLIT[dev/${_arch}] ${fname} (passthrough)"
        VERBATIM
      )
      list(APPEND _arch_dev_objs ${DEV_OBJ})
      list(APPEND _arch_kernel_fnames "${fname}")
      list(APPEND _arch_kernel_srcs "${src}")
    endforeach()

    # =======================================================================
    # Link all device objects for this arch
    # =======================================================================
    set(COMBINED_DEV_OBJ "${DEV_DIR}/combined.${_arch}.o")
    set(COMBINED_DEV_SO  "${DEV_DIR}/combined.${_arch}.so")

    list(LENGTH _arch_dev_objs _n_dev_objs)
    add_custom_command(
      OUTPUT  ${COMBINED_DEV_OBJ}
      COMMAND ${LLD} -r -o ${COMBINED_DEV_OBJ} ${_arch_dev_objs}
      DEPENDS ${_arch_dev_objs}
      COMMENT "SPLIT[link/${_arch}] combining ${_n_dev_objs} device objects"
      VERBATIM
    )

    add_custom_command(
      OUTPUT  ${COMBINED_DEV_SO}
      COMMAND ${LLD} -shared -o ${COMBINED_DEV_SO} ${COMBINED_DEV_OBJ}
      DEPENDS ${COMBINED_DEV_OBJ}
      COMMENT "SPLIT[cobj/${_arch}] producing code object"
      VERBATIM
    )

    list(APPEND _all_combined_sos ${COMBINED_DEV_SO})
  endforeach() # end per-arch loop

  # =========================================================================
  # Bundle all arches into hipfb
  # =========================================================================
  file(MAKE_DIRECTORY ${FAT_DIR} ${HOST_DIR})

  # Build bundler targets and inputs lists: host + one device per arch
  set(_bundler_targets "${SSC_BUNDLER_HOST_TARGET}")
  set(_bundler_inputs  "--input=/dev/null")
  foreach(_arch_so _dev_target IN ZIP_LISTS _all_combined_sos SSC_BUNDLER_DEVICE_TARGETS)
    string(APPEND _bundler_targets ",${_dev_target}")
    list(APPEND _bundler_inputs "--input=${_arch_so}")
  endforeach()

  set(COMBINED_HIPFB "${FAT_DIR}/combined.hipfb")
  set(_bundler_compress "")
  if(SSC_ENABLE_COMPRESS)
    set(_bundler_compress "--compress")
  endif()
  add_custom_command(
    OUTPUT  ${COMBINED_HIPFB}
    COMMAND ${BUNDLER}
      --type=bc
      --targets=${_bundler_targets}
      ${_bundler_inputs}
      --output=${COMBINED_HIPFB}
      ${_bundler_compress}
    DEPENDS ${_all_combined_sos}
    COMMENT "SPLIT[hipfb] creating fat binary blob (${_n_archs} arch(es))"
    VERBATIM
  )

  # =========================================================================
  # Host stubs for each kernel TU (dispatcher + passthrough)
  # Uses the first arch's kernel lists (same TUs across all arches)
  # =========================================================================

  # Collect kernel fnames/srcs from dispatcher + passthrough (arch-independent)
  set(_kernel_fnames "")
  set(_kernel_srcs "")
  foreach(src ${SSC_DISPATCHER_SOURCES})
    get_filename_component(fname ${src} NAME_WE)
    list(APPEND _kernel_fnames "${fname}")
    list(APPEND _kernel_srcs "${src}")
  endforeach()
  foreach(src ${SSC_PASSTHROUGH_KERNEL_SOURCES})
    get_filename_component(fname ${src} NAME_WE)
    list(APPEND _kernel_fnames "${fname}")
    list(APPEND _kernel_srcs "${src}")
  endforeach()

  # Build --offload-arch flags for all arches
  set(_offload_arch_flags "")
  foreach(_arch ${SSC_GPU_ARCHS})
    list(APPEND _offload_arch_flags "--offload-arch=${_arch}")
  endforeach()

  set(_kernel_host_objs "")
  list(LENGTH _kernel_srcs _n_kernel_srcs)
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
        ${_offload_arch_flags}
        -Xclang -fcuda-include-gpubinary
        -Xclang ${COMBINED_HIPFB}
        -c -O3 -fPIC
        ${_def_flags}
        ${_inc_flags}
        ${_fwd_opts}
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

  # Combine host stubs into a single object
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

  # Aggregate target
  add_custom_target(split_specialized_device DEPENDS ${COMBINED_FAT_OBJ})

  # Export the fat object to parent scope
  set(SPLIT_SPECIALIZED_FAT_OBJ ${COMBINED_FAT_OBJ} PARENT_SCOPE)

  message(STATUS "Split specialized: ${_n_spec} specialized + ${_n_kernel_srcs} kernel TUs × ${_n_archs} arch(es) → 1 fat object")
endfunction()
