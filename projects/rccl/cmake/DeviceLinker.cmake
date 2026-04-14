# cmake/DeviceLinker.cmake
#
# Assembly-extract device linker pipeline.
# Replaces the standard -fgpu-rdc compilation and linking of device code.
#
# All commands are expressed as add_custom_command so that the build system
# (ninja/make) can schedule them optimally alongside the rest of the build.
#
# Pipeline per specialized kernel (860 files, fully parallel, per GPU target):
#   compile .cpp -> .s  ->  extract .s -> extracted.s + .json  ->  assemble .o
#
# Dispatcher (per GPU target):
#   compile common.cu.cpp -> .s
#   aggregate resource .json files -> max_resources.json
#   patch dispatcher .s with max resources -> patched.s
#   assemble patched.s -> common_device.o
#
# Final:
#   lld -shared  ->  device.elf  (one per GPU target)
#   clang-offload-bundler  ->  device.hipfb  (bundles all GPU targets)
#   host compile common.cu.cpp with -fcuda-include-gpubinary  ->  common.o (fat)
#   normal HIP compile onerank.cu.cpp  ->  onerank.o (fat)
#
# Required variables (set by including CMakeLists.txt):
#   HIPIFY_DIR, GEN_DIR, GPU_TARGETS, PROJECT_BINARY_DIR, PROJECT_SOURCE_DIR,
#   ROCM_PATH, Python3_EXECUTABLE

message(STATUS "Device Linker: assembly-extract pipeline enabled")

set(DEVICE_BUILD_DIR "${PROJECT_BINARY_DIR}/device_build")
set(ASM_EXTRACT_DIR  "${PROJECT_SOURCE_DIR}/tools/asm_extract")
set(SPECIALIZED_DIR  "${GEN_DIR}/specialized")

# Derive tool paths from the C++ compiler cmake already resolved.
# ROCM_PATH may be empty in super-project builds (e.g. TheRock).
get_filename_component(_dl_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
find_program(DL_CLANG NAMES amdclang++ clang++ HINTS "${_dl_compiler_dir}" "${ROCM_PATH}/bin" REQUIRED)
find_program(DL_LLD   NAMES ld.lld             HINTS "${_dl_compiler_dir}" "${_dl_compiler_dir}/../lib/llvm/bin" "${ROCM_PATH}/llvm/bin" REQUIRED)
find_program(DL_BUNDLER NAMES clang-offload-bundler HINTS "${_dl_compiler_dir}" "${_dl_compiler_dir}/../lib/llvm/bin" "${ROCM_PATH}/llvm/bin" REQUIRED)

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
# Compile definitions — read from the rccl target (set by src/CMakeLists.txt)
# plus infrastructure defs that come from directory scope / linked targets.
# ---------------------------------------------------------------------------
get_target_property(_rccl_defs rccl COMPILE_DEFINITIONS)
set(DL_COMPILE_DEFS "")
if(_rccl_defs)
  foreach(_def ${_rccl_defs})
    list(APPEND DL_COMPILE_DEFS "-D${_def}")
  endforeach()
endif()

# Directory-scope defs (add_compile_definitions / add_definitions in root CMakeLists.txt)
# and defs inherited from linked targets (hip::device) that custom commands don't see.
list(APPEND DL_COMPILE_DEFS
  -DFMT_HEADER_ONLY=1
  -DNCCL_MAJOR=${NCCL_MAJOR}
  -DNCCL_MINOR=${NCCL_MINOR}
  -DNCCL_PATCH=${NCCL_PATCH}
  -DNCCL_VERSION_CODE=${NCCL_VERSION}
  -DROCM_VERSION=${ROCM_VERSION}
  -D__HIP_PLATFORM_AMD__=1
)

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
set(DL_INCLUDE_DIRS
  -I${PROJECT_BINARY_DIR}/include
  -I${HIPIFY_DIR}/src
  -I${HIPIFY_DIR}/src/device
  -I${HIPIFY_DIR}/src/device/network/unpack
  -I${HIPIFY_DIR}/src/include
  -I${HIPIFY_DIR}/src/include/mlx5
  -I${HIPIFY_DIR}/src/include/nccl_device
  -I${HIPIFY_DIR}/src/include/ionic
  -I${HIPIFY_DIR}/src/include/plugin
  -I${GEN_DIR}
  -isystem${ROCM_PATH}/include
)

# fmt is needed by proxy_trace.h (included transitively from collectives.cc).
# Only add the include path for FetchContent-fetched fmt; system-installed
# fmt headers are already in the compiler's default include path, and adding
# them explicitly via -isystem breaks #include_next ordering in device-only
# compilation (e.g., GCC's cmath can no longer find math.h).
if(fmt_SOURCE_DIR)
  list(APPEND DL_INCLUDE_DIRS -isystem${fmt_SOURCE_DIR}/include)
endif()

# ---------------------------------------------------------------------------
# Optimization / common flags
# ---------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(DL_OPT_FLAGS -O0 -g)
else()
  set(DL_OPT_FLAGS -O3)
endif()

# ===========================================================================
# Read specialized file list
# ===========================================================================
set(SPECIALIZED_FILES_TXT "${GEN_DIR}/specialized_files.txt")
if(NOT EXISTS "${SPECIALIZED_FILES_TXT}")
  message(FATAL_ERROR "Device Linker: ${SPECIALIZED_FILES_TXT} not found. generate.py must run first.")
endif()

file(STRINGS "${SPECIALIZED_FILES_TXT}" SPECIALIZED_ENTRIES)
list(LENGTH SPECIALIZED_ENTRIES DL_KERNEL_COUNT)
message(STATUS "Device Linker: ${DL_KERNEL_COUNT} specialized kernels")

# ===========================================================================
# Guard evaluation: skip kernels whose #if guard excludes this GPU target.
# Guards in specialized_files.txt look like:
#   (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)) && defined(ENABLE_LL128)
# ===========================================================================
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

# ===========================================================================
# Per-GPU-target device pipeline
# ===========================================================================
set(ALL_DEVICE_ELFS "")
set(DL_BUNDLER_TARGETS "host-x86_64-unknown-linux-gnu-")
set(DL_BUNDLER_INPUTS "--input=/dev/null")
set(ALL_IR_FILES "")

foreach(DL_GPU_TARGET ${DL_GPU_TARGETS})
  # CMake's Ninja generator sorts build rules by output path.  Without this,
  # "gfx1030" < "gfx906" lexicographically, so all RDNA kernels get lower
  # edge IDs and run first.  CDNA FP8 tree_simple kernels take ~6 minutes
  # each to compile; delaying them behind ~140s of RDNA work extends the
  # build tail significantly.  Prefix CDNA dirs with '-' ('-' < 'g') so
  # they sort first and those long compilations start immediately.
  if(DL_GPU_TARGET MATCHES "^gfx9")
    set(DL_ARCH_DIR "${DEVICE_BUILD_DIR}/-${DL_GPU_TARGET}")
  else()
    set(DL_ARCH_DIR "${DEVICE_BUILD_DIR}/${DL_GPU_TARGET}")
  endif()

  file(MAKE_DIRECTORY
    ${DL_ARCH_DIR}/specialized_asm
    ${DL_ARCH_DIR}/extracted_asm
    ${DL_ARCH_DIR}/extracted_obj
    ${DL_ARCH_DIR}/resources
  )

  set(DL_DEVICE_COMPILE_FLAGS
    -x hip --cuda-device-only --offload-arch=${DL_GPU_TARGET}
    --no-gpu-bundle-output
    -gline-tables-only
    -std=c++17
    -w
    ${DL_OPT_FLAGS}
  )

  # =========================================================================
  # Per-specialized-kernel commands (compile -> extract -> assemble)
  # =========================================================================
  set(ARCH_EXTRACTED_OBJS "")
  set(ARCH_RESOURCE_JSONS "")
  set(_dl_skipped 0)

  foreach(ENTRY ${SPECIALIZED_ENTRIES})
    # Format: "filename funcname [guard]"
    if(NOT ENTRY MATCHES "^([^ ]+) +([^ ]+) *(.*)")
      continue()
    endif()
    set(CPP_FILE "${CMAKE_MATCH_1}")
    set(_entry_guard "${CMAKE_MATCH_3}")
    string(REGEX REPLACE "\\.cpp$" "" BASE "${CPP_FILE}")

    dl_evaluate_guard("${_entry_guard}" "${DL_GPU_TARGET}" _guard_ok)
    if(NOT _guard_ok)
      math(EXPR _dl_skipped "${_dl_skipped} + 1")
      continue()
    endif()

    set(SRC      "${SPECIALIZED_DIR}/${CPP_FILE}")
    set(ASM_OUT  "${DL_ARCH_DIR}/specialized_asm/${BASE}.s")
    set(EXT_ASM  "${DL_ARCH_DIR}/extracted_asm/${BASE}.s")
    set(RES_JSON "${DL_ARCH_DIR}/resources/${BASE}.json")
    set(OBJ_OUT  "${DL_ARCH_DIR}/extracted_obj/${BASE}.o")

    # Step 1: Compile specialized kernel to assembly
    add_custom_command(
      OUTPUT  ${ASM_OUT}
      COMMAND ${DL_CLANG}
        -DRCCL_DEVICE_LINKER
        ${DL_COMPILE_DEFS}
        ${DL_INCLUDE_DIRS}
        ${DL_DEVICE_COMPILE_FLAGS}
        -S
        -o ${ASM_OUT}
        ${SRC}
      DEPENDS ${SRC}
      COMMENT "DL [${DL_GPU_TARGET}] compile: ${CPP_FILE}"
      VERBATIM
    )

    # Step 2: Extract device function + resource usage
    add_custom_command(
      OUTPUT  ${EXT_ASM} ${RES_JSON}
      COMMAND ${Python3_EXECUTABLE} ${ASM_EXTRACT_DIR}/extract_device_function.py
        ${ASM_OUT} ${EXT_ASM} ${RES_JSON}
        > /dev/null
      DEPENDS ${ASM_OUT} ${ASM_EXTRACT_DIR}/extract_device_function.py
      COMMENT "DL [${DL_GPU_TARGET}] extract: ${BASE}"
      VERBATIM
    )

    # Step 3: Assemble extracted function to relocatable object
    add_custom_command(
      OUTPUT  ${OBJ_OUT}
      COMMAND ${DL_CLANG}
        -x assembler -target amdgcn-amd-amdhsa -mcpu=${DL_GPU_TARGET}
        -c -o ${OBJ_OUT}
        ${EXT_ASM}
      DEPENDS ${EXT_ASM}
      COMMENT "DL [${DL_GPU_TARGET}] assemble: ${BASE}"
      VERBATIM
    )

    list(APPEND ARCH_EXTRACTED_OBJS ${OBJ_OUT})
    list(APPEND ARCH_RESOURCE_JSONS ${RES_JSON})
  endforeach()

  list(LENGTH ARCH_EXTRACTED_OBJS _dl_built)
  if(_dl_skipped GREATER 0)
    message(STATUS "Device Linker [${DL_GPU_TARGET}]: ${_dl_built} kernels to build, ${_dl_skipped} skipped (arch guard)")
  endif()

  # =========================================================================
  # Dispatcher: compile common.cu.cpp to device assembly
  # =========================================================================
  set(ARCH_COMMON_DEVICE_ASM "${DL_ARCH_DIR}/common_device.s")

  add_custom_command(
    OUTPUT  ${ARCH_COMMON_DEVICE_ASM}
    COMMAND ${DL_CLANG}
      -DRCCL_DEVICE_LINKER
      -DUSE_INDIRECT_FUNCTION_CALL
      ${DL_COMPILE_DEFS}
      ${DL_INCLUDE_DIRS}
      -x hip --cuda-device-only --offload-arch=${DL_GPU_TARGET}
      --no-gpu-bundle-output
      -g
      -std=c++17
      -w
      ${DL_OPT_FLAGS}
      -S
      -o ${ARCH_COMMON_DEVICE_ASM}
      ${HIPIFY_DIR}/src/device/common.cu.cpp
    DEPENDS ${HIPIFY_DIR}/src/device/common.cu.cpp
    COMMENT "DL [${DL_GPU_TARGET}] compile dispatcher: common.cu.cpp -> assembly"
    VERBATIM
  )

  # =========================================================================
  # Aggregate resource usage across all specialized functions
  # =========================================================================
  set(ARCH_MAX_RESOURCES_JSON "${DL_ARCH_DIR}/max_resources.json")

  add_custom_command(
    OUTPUT  ${ARCH_MAX_RESOURCES_JSON}
    COMMAND ${Python3_EXECUTABLE} ${ASM_EXTRACT_DIR}/aggregate_resources.py
      ${DL_ARCH_DIR}/resources
      ${ARCH_MAX_RESOURCES_JSON}
      ${DL_GPU_TARGET}
    DEPENDS ${ARCH_RESOURCE_JSONS} ${ASM_EXTRACT_DIR}/aggregate_resources.py
    COMMENT "DL [${DL_GPU_TARGET}] aggregate: resource usage from ${_dl_built} functions"
    VERBATIM
  )

  # =========================================================================
  # Patch dispatcher assembly with aggregated resource values
  # =========================================================================
  set(ARCH_COMMON_DEVICE_PATCHED "${DL_ARCH_DIR}/common_device_patched.s")

  add_custom_command(
    OUTPUT  ${ARCH_COMMON_DEVICE_PATCHED}
    COMMAND ${Python3_EXECUTABLE} ${ASM_EXTRACT_DIR}/patch_dispatcher.py
      ${ARCH_COMMON_DEVICE_ASM}
      ${ARCH_COMMON_DEVICE_PATCHED}
      ${ARCH_MAX_RESOURCES_JSON}
      ${DL_GPU_TARGET}
    DEPENDS ${ARCH_COMMON_DEVICE_ASM} ${ARCH_MAX_RESOURCES_JSON}
            ${ASM_EXTRACT_DIR}/patch_dispatcher.py
    COMMENT "DL [${DL_GPU_TARGET}] patch dispatcher with max resources"
    VERBATIM
  )

  # =========================================================================
  # Assemble patched dispatcher
  # =========================================================================
  set(ARCH_COMMON_DEVICE_OBJ "${DL_ARCH_DIR}/common_device.o")

  add_custom_command(
    OUTPUT  ${ARCH_COMMON_DEVICE_OBJ}
    COMMAND ${DL_CLANG}
      -x assembler -target amdgcn-amd-amdhsa -mcpu=${DL_GPU_TARGET}
      -c -o ${ARCH_COMMON_DEVICE_OBJ}
      ${ARCH_COMMON_DEVICE_PATCHED}
    DEPENDS ${ARCH_COMMON_DEVICE_PATCHED}
    COMMENT "DL [${DL_GPU_TARGET}] assemble: common_device.o"
    VERBATIM
  )

  # =========================================================================
  # Link all device objects into device.elf for this architecture
  # =========================================================================
  set(ARCH_DEVICE_ELF "${DL_ARCH_DIR}/device.elf")
  set(ARCH_LINK_RSP   "${DL_ARCH_DIR}/device_link.rsp")

  list(JOIN ARCH_EXTRACTED_OBJS "\n" _arch_objs_newline)
  file(GENERATE OUTPUT ${ARCH_LINK_RSP}
    CONTENT "${ARCH_COMMON_DEVICE_OBJ}\n${_arch_objs_newline}\n")

  add_custom_command(
    OUTPUT  ${ARCH_DEVICE_ELF}
    COMMAND ${DL_LLD} -shared
      -o ${ARCH_DEVICE_ELF}
      @${ARCH_LINK_RSP}
    DEPENDS ${ARCH_COMMON_DEVICE_OBJ} ${ARCH_EXTRACTED_OBJS}
    COMMENT "DL [${DL_GPU_TARGET}] link: device.elf"
    VERBATIM
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
        ${DL_COMPILE_DEFS}
        ${DL_INCLUDE_DIRS}
        ${DL_DEVICE_COMPILE_FLAGS}
        -emit-llvm -S
        -o ${IR_OUT}
        ${SRC}
      DEPENDS ${SRC}
      COMMENT "DL [${DL_GPU_TARGET}] IR: ${CPP_FILE}"
      VERBATIM
    )
    list(APPEND ALL_IR_FILES ${IR_OUT})
  endforeach()

endforeach()  # end of per-GPU-target loop

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

add_custom_command(
  OUTPUT  ${COMMON_FAT_OBJ}
  COMMAND ${DL_CLANG}
    -x hip --offload-host-only ${DL_OFFLOAD_ARCH_FLAGS}
    -Xclang -fcuda-include-gpubinary -Xclang ${DEVICE_HIPFB}
    -DRCCL_DEVICE_LINKER
    -DUSE_INDIRECT_FUNCTION_CALL
    ${DL_COMPILE_DEFS}
    ${DL_INCLUDE_DIRS}
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
    -DRCCL_DEVICE_LINKER
    ${DL_COMPILE_DEFS}
    ${DL_INCLUDE_DIRS}
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
    -DRCCL_DEVICE_LINKER
    ${DL_COMPILE_DEFS}
    ${DL_INCLUDE_DIRS}
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
