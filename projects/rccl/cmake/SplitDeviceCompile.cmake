# cmake/SplitDeviceCompile.cmake
#
# Split device compilation pipeline for RCCL.
#
# Instead of the standard -fgpu-rdc flow (which defers all backend passes to
# link time, serialising them), this module pre-compiles each device TU through
# the full LLVM backend independently and in parallel:
#
#   source.cpp -> LLVM bitcode (.bc) -> device ELF (.o) via llc
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
  cmake_parse_arguments(SDC "" "TARGET;GPU_ARCH;ROCM_PATH;OUTPUT_DIR" "SOURCES;FUNC_ONLY_SOURCES;INCLUDE_DIRS;COMPILE_DEFS" ${ARGN})

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

  set(LLC     "${SDC_ROCM_PATH}/llvm/bin/llc")
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

  set(ALL_FAT_OBJECTS "")

  list(LENGTH SDC_SOURCES _n_sources)
  message(STATUS "Split device compile: ${_n_sources} TUs for ${SDC_GPU_ARCH}")

  foreach(src ${SDC_SOURCES})
    get_filename_component(fname ${src} NAME_WE)

    # Determine if this source needs -DNCCL_FUNC_ONLY
    set(_extra_defs "")
    list(FIND SDC_FUNC_ONLY_SOURCES "${src}" _func_only_idx)
    if(NOT _func_only_idx EQUAL -1)
      set(_extra_defs "-DNCCL_FUNC_ONLY")
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
        -fvisibility=hidden
        -Wno-unused-function
        -Wno-format-nonliteral
        -o ${BC_FILE} ${src}
      DEPENDS   ${src}
      COMMENT   "SPLIT[bc]  ${fname}"
      VERBATIM
    )

    # -- Step B: LLC to relocatable device ELF object -------------------------
    add_custom_command(
      OUTPUT  ${DEV_OBJ}
      COMMAND ${LLC}
        -O3
        -mtriple=amdgcn-amd-amdhsa
        -mcpu=${SDC_GPU_ARCH}
        -filetype=obj
        -o ${DEV_OBJ} ${BC_FILE}
      DEPENDS   ${BC_FILE}
      COMMENT   "SPLIT[llc] ${fname}"
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
        --targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--${SDC_GPU_ARCH}
        --input=${HOST_OBJ}
        --input=${DEV_OBJ}
        --output=${FAT_OBJ}
      DEPENDS   ${HOST_OBJ} ${DEV_OBJ}
      COMMENT   "SPLIT[fat]  ${fname}"
      VERBATIM
    )

    list(APPEND ALL_FAT_OBJECTS ${FAT_OBJ})
  endforeach()

  # Aggregate target so the build system can build all TUs in parallel
  add_custom_target(rccl_device_objects DEPENDS ${ALL_FAT_OBJECTS})

  # Export the list of fat objects to the parent scope
  set(DEVICE_FAT_OBJECTS ${ALL_FAT_OBJECTS} PARENT_SCOPE)

  message(STATUS "Split device compile: ${_n_sources} fat objects will be produced in ${FAT_DIR}")
endfunction()
