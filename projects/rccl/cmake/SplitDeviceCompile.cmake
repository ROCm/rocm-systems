# cmake/SplitDeviceCompile.cmake
#
# Split device compilation pipeline for RCCL.
#
# Instead of the standard -fgpu-rdc flow (which defers all backend passes to
# link time, serialising them), this module pre-compiles each device TU through
# the full LLVM backend independently and in parallel:
#
#   source.cpp -> LLVM bitcode (.bc) -> device ELF (.o) via clang -x ir
#                                          |
#   source.cpp -> host stub (.host.o) -----+
#                                          v
#                               fat object (.fat.o) via clang-offload-bundler
#
# The fat objects are then linked with `amdclang++ -fgpu-rdc --hip-link` which
# only performs lightweight device-side symbol resolution (~1-2 seconds) instead
# of re-running the entire backend.
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

  # Output directories
  set(BC_DIR   "${SDC_OUTPUT_DIR}/split_device/bc")
  set(DEV_DIR  "${SDC_OUTPUT_DIR}/split_device/dev_obj")
  set(HOST_DIR "${SDC_OUTPUT_DIR}/split_device/host_obj")
  set(FAT_DIR  "${SDC_OUTPUT_DIR}/split_device/fat_obj")
  file(MAKE_DIRECTORY ${BC_DIR} ${DEV_DIR} ${HOST_DIR} ${FAT_DIR})

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
  set(ALL_DEV_OBJECTS "")
  set(_kernel_dev_obj "")
  set(_kernel_host_obj "")
  set(_kernel_fat_obj "")
  set(_kernel_fname "")

  list(LENGTH SDC_SOURCES _n_sources)
  message(STATUS "Split device compile: ${_n_sources} TUs for ${SDC_GPU_ARCH}")

  foreach(src ${SDC_SOURCES})
    get_filename_component(fname ${src} NAME_WE)

    # Determine if this source needs -DNCCL_FUNC_ONLY
    set(_extra_defs "")
    set(_is_kernel_tu FALSE)
    list(FIND SDC_FUNC_ONLY_SOURCES "${src}" _func_only_idx)
    if(NOT _func_only_idx EQUAL -1)
      set(_extra_defs "-DNCCL_FUNC_ONLY")
    else()
      set(_is_kernel_tu TRUE)
    endif()

    set(BC_FILE  "${BC_DIR}/${fname}.${SDC_GPU_ARCH}.bc")
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

    # -- Step B: Compile bitcode to relocatable device ELF object -------------
    # Use amdclang++ -x ir instead of standalone llc so the driver
    # automatically applies all target-specific backend flags (target features,
    # AMDGPU options, etc.) that vary across ROCm versions.
    add_custom_command(
      OUTPUT  ${DEV_OBJ}
      COMMAND ${CMAKE_CXX_COMPILER}
        -x ir
        -target amdgcn-amd-amdhsa
        -mcpu=${SDC_GPU_ARCH}
        -O3 -c
        -o ${DEV_OBJ} ${BC_FILE}
      DEPENDS   ${BC_FILE}
      COMMENT   "SPLIT[dev] ${fname}"
      VERBATIM
    )

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

    list(APPEND ALL_DEV_OBJECTS ${DEV_OBJ})

    if(_is_kernel_tu)
      # Defer Step D for the kernel TU until after the KD patch step below.
      set(_kernel_dev_obj  ${DEV_OBJ})
      set(_kernel_host_obj ${HOST_OBJ})
      set(_kernel_fat_obj  ${FAT_OBJ})
      set(_kernel_fname    ${fname})
    else()
      # -- Step D: Bundle host + device into fat object -----------------------
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
    endif()

    list(APPEND ALL_FAT_OBJECTS ${FAT_OBJ})
  endforeach()

  # -- Step D-kernel: Patch kernel descriptors, then bundle ------------------
  # The kernel TU (common.cu) dispatches device functions through a function
  # pointer table (indirect calls).  The compiler cannot propagate callee
  # VGPR/scratch requirements through the indirection, so the kernel
  # descriptor's granulated_workitem_vgpr_count and private_segment_fixed_size
  # may be too low.  We fix this by analysing all compiled device-function
  # objects and patching the kernel descriptor before bundling.
  if(_kernel_dev_obj)
    set(_patch_stamp "${SDC_OUTPUT_DIR}/split_device/.kd_patched")
    set(_patch_script "${CMAKE_SOURCE_DIR}/cmake/scripts/patch_kernel_descriptor.py")

    add_custom_command(
      OUTPUT  ${_patch_stamp}
      COMMAND ${Python3_EXECUTABLE} ${_patch_script}
        --kernel-obj ${_kernel_dev_obj}
        --dev-obj-dir ${DEV_DIR}
        --gpu-arch ${SDC_GPU_ARCH}
        --rocm-path ${SDC_ROCM_PATH}
      COMMAND ${CMAKE_COMMAND} -E touch ${_patch_stamp}
      DEPENDS   ${ALL_DEV_OBJECTS}
      COMMENT   "SPLIT[patch] Fixing kernel descriptors for ${_kernel_fname}"
      VERBATIM
    )

    add_custom_command(
      OUTPUT  ${_kernel_fat_obj}
      COMMAND ${BUNDLER}
        --type=o
        --targets=${SDC_BUNDLER_HOST_TARGET},${SDC_BUNDLER_DEVICE_TARGET}
        --input=${_kernel_host_obj}
        --input=${_kernel_dev_obj}
        --output=${_kernel_fat_obj}
      DEPENDS   ${_kernel_host_obj} ${_kernel_dev_obj} ${_patch_stamp}
      COMMENT   "SPLIT[fat]  ${_kernel_fname}"
      VERBATIM
    )
  endif()

  # Aggregate target so the build system can build all TUs in parallel
  add_custom_target(rccl_device_objects DEPENDS ${ALL_FAT_OBJECTS})

  # Export the list of fat objects to the parent scope
  set(DEVICE_FAT_OBJECTS ${ALL_FAT_OBJECTS} PARENT_SCOPE)

  message(STATUS "Split device compile: ${_n_sources} fat objects will be produced in ${FAT_DIR}")
endfunction()
