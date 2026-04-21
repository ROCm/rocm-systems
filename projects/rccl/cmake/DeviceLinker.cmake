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

# Tell the driver where to find the real compiler.
get_filename_component(_dl_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
find_program(DL_CLANG NAMES amdclang++ clang++
  HINTS "${_dl_compiler_dir}" "${ROCM_PATH}/bin" REQUIRED)
find_program(DL_BUNDLER NAMES clang-offload-bundler
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
  )
  target_compile_definitions(${_dev_target} PRIVATE RCCL_DEVICE_LINKER)
  target_link_libraries(${_dev_target} PRIVATE rccl_device_defs)

  add_dependencies(${_dev_target} hipify_all)

  # =========================================================================
  # Link step: driver --link mode produces device.elf
  # =========================================================================
  set(ARCH_DEVICE_ELF "${DL_ARCH_DIR}/device.elf")

  # Gather definitions and includes for the dispatcher compilation inside --link.
  # The driver forwards these to amdclang++ when compiling common.cu.cpp.
  get_target_property(_dev_defs ${_dev_target} COMPILE_DEFINITIONS)
  set(_link_def_flags "")
  if(_dev_defs)
    foreach(_d ${_dev_defs})
      list(APPEND _link_def_flags "-D${_d}")
    endforeach()
  endif()
  # Also add interface definitions from rccl_device_defs
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

  set(_link_rsp "${DL_ARCH_DIR}/link_objects.rsp")
  file(GENERATE OUTPUT "${_link_rsp}"
    CONTENT "$<JOIN:$<TARGET_OBJECTS:${_dev_target}>,\n>\n")

  add_custom_command(
    OUTPUT  ${ARCH_DEVICE_ELF}
    COMMAND ${CMAKE_RCCLDEV_COMPILER}
      --link
      --arch=${DL_GPU_TARGET}
      --clang=${DL_CLANG}
      ${DL_HIP_COMPILER_FLAGS}
      --dispatcher=${HIPIFY_DIR}/src/device/common.cu.cpp
      ${_link_def_flags}
      ${_link_inc_flags}
      ${DL_OPT_FLAGS}
      -std=c++17
      -o ${ARCH_DEVICE_ELF}
      @${_link_rsp}
    DEPENDS ${_dev_target} ${HIPIFY_DIR}/src/device/common.cu.cpp
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
        -std=c++17 -w ${DL_OPT_FLAGS}
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
  COMMENT "DL bundle: device.elf(s) -> device.hipfb [${DL_GPU_TARGETS}]"
  VERBATIM
)

# ===========================================================================
# Host compile common.cu.cpp with embedded device binary
# ===========================================================================
set(COMMON_FAT_OBJ "${DEVICE_BUILD_DIR}/common.o")

set(DL_HOST_COMPRESS "")
if(ENABLE_COMPRESS)
  set(DL_HOST_COMPRESS "--offload-compress")
endif()

# Gather include flags for host compile (same paths as device)
set(_host_inc_flags "")
if(_dev_includes)
  foreach(_inc ${_dev_includes})
    list(APPEND _host_inc_flags "-I${_inc}")
  endforeach()
endif()
if(_dev_sys_includes)
  foreach(_inc ${_dev_sys_includes})
    if(NOT _inc IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
      list(APPEND _host_inc_flags "-isystem${_inc}")
    endif()
  endforeach()
endif()

add_custom_command(
  OUTPUT  ${COMMON_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip --offload-host-only ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -Xclang -fcuda-include-gpubinary -Xclang ${DEVICE_HIPFB}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    -std=c++17
    -fPIC
    -w
    ${DL_HOST_COMPRESS}
    -c -o ${COMMON_FAT_OBJ}
    ${HIPIFY_DIR}/src/device/common.cu.cpp
  DEPENDS ${DEVICE_HIPFB} ${HIPIFY_DIR}/src/device/common.cu.cpp
  COMMENT "DL host compile: common.cu.cpp with embedded device binary"
  VERBATIM
)

# ===========================================================================
# Onerank: normal HIP compilation (host+device, no RDC)
# ===========================================================================
set(ONERANK_FAT_OBJ "${DEVICE_BUILD_DIR}/onerank.o")

add_custom_command(
  OUTPUT  ${ONERANK_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${ONERANK_FAT_OBJ}
    ${HIPIFY_DIR}/src/device/onerank.cu.cpp
  DEPENDS ${HIPIFY_DIR}/src/device/onerank.cu.cpp
  COMMENT "DL compile: onerank.cu.cpp (normal fat object)"
  VERBATIM
)

# ===========================================================================
# collectives.cc: contains a __global__ kernel launch (hierarchicalAGShuffle)
# so it needs full HIP compilation, not --offload-host-only.
# ===========================================================================
set(COLLECTIVES_FAT_OBJ "${DEVICE_BUILD_DIR}/collectives.o")

add_custom_command(
  OUTPUT  ${COLLECTIVES_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip ${DL_OFFLOAD_ARCH_FLAGS}
    ${DL_HIP_COMPILER_FLAGS}
    -DRCCL_DEVICE_LINKER
    ${_link_def_flags}
    ${_host_inc_flags}
    ${DL_OPT_FLAGS}
    -std=c++17
    -fPIC
    -w
    -c -o ${COLLECTIVES_FAT_OBJ}
    ${HIPIFY_DIR}/src/collectives.cc
  DEPENDS ${HIPIFY_DIR}/src/collectives.cc
  COMMENT "DL compile: collectives.cc (has __global__ kernel)"
  VERBATIM
)

# ===========================================================================
# Top-level target
# ===========================================================================
add_custom_target(device_linker_build ALL
  DEPENDS ${COMMON_FAT_OBJ} ${ONERANK_FAT_OBJ} ${COLLECTIVES_FAT_OBJ}
)
add_dependencies(device_linker_build hipify_all)

set(DEVICE_LINKER_OBJECTS
  ${COMMON_FAT_OBJ}
  ${ONERANK_FAT_OBJ}
  ${COLLECTIVES_FAT_OBJ}
)

# ===========================================================================
# Optional: emit LLVM IR (ninja device_ir)
# ===========================================================================
add_custom_target(device_ir DEPENDS ${ALL_IR_FILES})
add_dependencies(device_ir hipify_all)
