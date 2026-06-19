#[[
# Run the scripts once during configuration to get the file lists
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    OUTPUT_VARIABLE files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${files}" files)
list(TRANSFORM files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E env NCCL_USE_CMAKE=1 ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    OUTPUT_VARIABLE symmetric_files
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
string(STRIP "${symmetric_files}" symmetric_files)
list(TRANSFORM symmetric_files PREPEND ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric/)

# On Windows, exclude GIN device sources (GIN backend implementation is Linux-only; would cause nvlink undefined refs).
set(device_files ${files})
set(device_symmetric_files ${symmetric_files})
if(NCCL_OS_WINDOWS)
  list(FILTER device_files EXCLUDE REGEX "gin")
  list(FILTER device_symmetric_files EXCLUDE REGEX "gin")
endif()

# Create custom commands to generate source files with proper dependencies
add_custom_command(
    OUTPUT  ${files}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating device source files"
)

add_custom_command(
    OUTPUT  ${symmetric_files}
    COMMAND ${CMAKE_COMMAND} -E env ${symmetric_gen_env} ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${CMAKE_CURRENT_BINARY_DIR}/gensrc/symmetric "${ONLY_FUNCS}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating symmetric device source files"
)

# On Linux, ncclGin_BackendMask<...> is provided by implicit instantiation in common.cu (no -G).
# On Windows, GIN device code is excluded (device_files / device_symmetric_files filtered above).

# On Windows use STATIC so CUDA device linking runs (OBJECT libs don't support it); on Linux use OBJECT.
if(NCCL_OS_WINDOWS)
    add_library(nccl_device STATIC
                ${device_files}
                ${device_symmetric_files}
                ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
    )
    set_target_properties(nccl_device PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
    )
else()
    add_library(nccl_device OBJECT
                ${files}
                ${symmetric_files}
                ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
    )
    set_target_properties(nccl_device PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
endif()


# On Linux, define NCCL_OS_LINUX so nccl_device.h includes gin__funcs.h (required for ncclGin_BackendMask<...> in common.cu and symmetric GIN kernels)
if(NCCL_OS_LINUX)
    target_compile_definitions(nccl_device PRIVATE NCCL_OS_LINUX)
endif()

# WIN32_LEAN_AND_MEAN prevents windows.h from pulling in winsock.h, so _WINSOCKAPI_
# is only defined once (by winsock2.h) and C4005 redefinition warnings are avoided.
# Skip GIN/DOCA (GDAKI) on Windows: doca_gpunetio headers are not available and GIN backend is Linux-only.
if(NCCL_OS_WINDOWS)
    target_compile_definitions(nccl_device PRIVATE WIN32_LEAN_AND_MEAN NCCL_GIN_GDAKI_ENABLE=0)
endif()

# Export-safe includes: only INSTALL_INTERFACE in PUBLIC; build paths in PRIVATE so export does not see source/build dirs.
target_include_directories(nccl_device PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(nccl_device PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_SOURCE_DIR}/src/include/plugin
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)

add_dependencies(nccl_device nccl_header)
]]


set(DEVICE_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/all_gather.h
  ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_v.h
  ${CMAKE_CURRENT_SOURCE_DIR}/all_reduce.h
  ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_pivot.h
  ${CMAKE_CURRENT_SOURCE_DIR}/alltoall_gda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/alltoallv_gda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/broadcast.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common_kernel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/op128.h
  ${CMAKE_CURRENT_SOURCE_DIR}/primitives.h
  ${CMAKE_CURRENT_SOURCE_DIR}/prims_ll128.h
  ${CMAKE_CURRENT_SOURCE_DIR}/prims_ll.h
  ${CMAKE_CURRENT_SOURCE_DIR}/prims_simple.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce_kernel.h
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce_scatter.h
  ${CMAKE_CURRENT_SOURCE_DIR}/rccl_metadata.h
  ${CMAKE_CURRENT_SOURCE_DIR}/sendrecv.h
  ${CMAKE_CURRENT_SOURCE_DIR}/hierarchical_shuffle.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ce_reduce.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/onerank.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/tdm/cachePolicy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tdm/syncPolicy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tdm/asyncCopy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tdm/tdmCopy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/network/unpack/unpack_defs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/network/unpack/unpack.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_gather.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_gather_gin.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/all_reduce.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/data_ops.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch__funcs.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/gin_scratch__types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/kernel.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/primitives.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/reduce_scatter.cuh
  ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/reduce_scatter_gin.cuh
)

#==================================================================================================
# RCCL EQUIVALENT -- mirrors the NCCL blocks above. Each block notes the NCCL lines it corresponds
# to and the RCCL origin it was COPIED from (originals remain live in those locations).
#==================================================================================================

# --- [NCCL L1-8 + L10-16] Configure-time device source generation -------------------------------
# NCCL runs generate.py / symmetric/generate.py at configure time and captures the emitted file
# list from stdout. RCCL runs the same generators but passes additional RCCL-only arguments
# (IFC_ENABLED, COLLTRACE, BUILD_LOCAL_GPU_TARGET_ONLY, ENABLE_ROCSHMEM, ONLY_FUNCS) and discovers
# the generated files by globbing afterwards (see next block) rather than parsing stdout.
# COPIED FROM: projects/rccl/src/CMakeLists.txt L586-624.

set(GEN_SYM_DIR "${GEN_DIR}/symmetric")
set(GEN_CE_REDUCE_DIR "${GEN_DIR}/ce_reduce")

if(ONLY_FUNCS)
  message(WARNING "Using ONLY_FUNCS = ${ONLY_FUNCS}. Not meant for release builds.")
endif()

# Execute the python script to generate required collective functions
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate.py ${GEN_DIR} ${IFC_ENABLED} OFF ${BUILD_LOCAL_GPU_TARGET_ONLY} ${ENABLE_ROCSHMEM} ${ONLY_FUNCS}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_py_result
    ERROR_VARIABLE gen_py_error
)
if (gen_py_result)
  message(SEND_ERROR "Error: ${gen_py_error}")
  message(FATAL_ERROR "${CMAKE_CURRENT_SOURCE_DIR}/generate.py failed")
endif()

if (GENERATE_SYM_KERNELS)
  # Execute the python script to generate required symmetric memory kernels
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py ${GEN_SYM_DIR}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_sym_py_result
    ERROR_VARIABLE gen_sym_py_error
  )
  if (gen_sym_py_result)
    message(SEND_ERROR "Error: ${gen_sym_py_error}")
    message(FATAL_ERROR "${CMAKE_CURRENT_SOURCE_DIR}/symmetric/generate.py failed")
  endif()
endif()

# Execute the python script to generate one TU per CE-reduce kernel
# instantiation (see src/device/ce_reduce/generate.py for why this is split
# out instead of living in a single ce_reduce.cc).
execute_process(
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ce_reduce/generate.py ${GEN_CE_REDUCE_DIR}
    WORKING_DIRECTORY ${RCCL_SOURCE_DIR}
    RESULT_VARIABLE gen_ce_reduce_py_result
    ERROR_VARIABLE gen_ce_reduce_py_error
)
if (gen_ce_reduce_py_result)
  message(SEND_ERROR "Error: ${gen_ce_reduce_py_error}")
  message(FATAL_ERROR "${CMAKE_CURRENT_SOURCE_DIR}/ce_reduce/generate.py failed")
endif()

# --- [NCCL L18-33] Collect generated sources ----------------------------------------------------
# NCCL declares add_custom_command rules (OUTPUT ${files} / ${symmetric_files}) so generated files
# rebuild on demand. RCCL has no build-time regeneration command; it globs the configure-time
# output and appends to HIP_SOURCES (the master hipified source list owned by src/CMakeLists.txt),
# then unconditionally drops gensrc/specialized/ (those belong only to the device-linker pipeline).
# COPIED FROM: projects/rccl/src/CMakeLists.txt L626-637.
# Find the generated files in the output directory
file(GLOB_RECURSE DEVICE_GEN_SOURCES "${GEN_DIR}/*")

# Specialized kernels are only used by the device linker pipeline;
# always exclude them from the normal HIP compilation to avoid
# duplicate symbols with the aggregate per-collective .cpp files.
list(FILTER DEVICE_GEN_SOURCES EXCLUDE REGEX "gensrc/specialized/")

# --- [NCCL L35-58] Build the device library / device objects ------------------------------------
# This is the heart of NCCL's file: add_library(nccl_device OBJECT files symmetric common.cu
# onerank.cu) + CUDA separable-compilation properties + include dirs + nccl_header dependency.
# RCCL has no standalone device library; the device translation units become part of librccl. The
# two RCCL paths below are selected by ENABLE_DEVICE_LINKER and together stand in for NCCL L35-58.
if(ENABLE_DEVICE_LINKER)
  # ON path == NCCL's "separately compiled + device-symbol-resolved device library" idea, realized
  # via the RCCL assembly-extract pipeline. The per-collective .cpp files and the kernel-bearing
  # TUs (common.cu, onerank.cu, collectives.cc, dda_*_ipc.cu) are removed from HIP_SOURCES because
  # DeviceLinker.cmake compiles/links them separately into DEVICE_LINKER_OBJECTS.
  # COPIED FROM: projects/rccl/src/CMakeLists.txt L639-658 (HIP_SOURCES filtering).
  # Remove per-collective .cpp files from gensrc/ and any subdirs (replaced by specialized/symmetric pipelines)
  list(FILTER DEVICE_GEN_SOURCES EXCLUDE REGEX "gensrc/.*\\.cpp$")
  # Add back host-only generated files
  list(APPEND DEVICE_GEN_SOURCES "${GEN_DIR}/host_table.cpp")
  if(EXISTS "${GEN_DIR}/device_table.cpp")
    list(APPEND DEVICE_GEN_SOURCES "${GEN_DIR}/device_table.cpp")
  endif()
  message(STATUS "Device Linker: filtered DEVICE_GEN_SOURCES (removed device .cpp under gensrc/ + subdirs, common.cu, onerank.cu, collectives.cc, dda_all_reduce_ipc.cu.cpp, dda_reduce_scatter_ipc.cu.cpp, dda_all_gather_ipc.cu.cpp, dda_alltoall_ipc.cu.cpp, dda_all_reduce_fabric.cu.cpp, dda_all_reduce_fabric_ll.cu.cpp, dda_reduce_scatter_fabric.cu.cpp, dda_all_gather_fabric.cu.cpp, dda_all_gather_fabric_ll.cu.cpp, dda_alltoall_fabric.cu.cpp)")

  set(DEVICE_REMOVED_SOURCES
    ${HIPIFY_SRC}/collectives.cc
    ${HIPIFY_SRC}/algorithms/dda/all_reduce/dda_all_reduce_ipc.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/reduce_scatter/dda_reduce_scatter_ipc.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/all_gather/dda_all_gather_ipc.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/alltoall/dda_alltoall_ipc.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/all_reduce/dda_all_reduce_fabric.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/all_reduce/dda_all_reduce_fabric_ll128.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/all_gather/dda_all_gather_fabric.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/all_gather/dda_all_gather_fabric_ll.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/all_gather/dda_all_gather_fabric_ll128.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/alltoall/dda_alltoall_fabric.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/alltoall/dda_alltoall_fabric_ll.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/alltoall/dda_alltoall_fabric_ll128.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll.cu.cpp
    ${HIPIFY_SRC}/algorithms/dda/reduce_scatter/dda_reduce_scatter_fabric_ll128.cu.cpp
    # ce_reduce.cc is now pure host code (a dispatcher calling into the
    # per-instantiation launchers generated under gensrc/ce_reduce/, which are
    # excluded above by the "gensrc/.*\\.cpp$" filter and compiled separately
    # by cmake/DeviceLinker.cmake). It stays in the main target and is
    # compiled --offload-host-only like ce_coll.cc.
    ${HIPIFY_SRC}/device/common.cu.cpp
    ${HIPIFY_SRC}/device/onerank.cu.cpp
  )
else()
  # OFF path (conventional): no separate device library. The generated device sources (appended to
  # HIP_SOURCES above) plus common.cu / onerank.cu (listed in src/CMakeLists.txt SRC_FILES) are
  # compiled directly into librccl as relocatable device code; device symbols are resolved at the
  # final librccl link. This is RCCL's analog of NCCL's CUDA_SEPARABLE_COMPILATION (NCCL L43-46).
  # COPIED FROM: projects/rccl/src/CMakeLists.txt L1034.
  set(DEVICE_REMOVED_SOURCES "")
endif()

# [NCCL L48-56] Include directories: NCCL sets them on the nccl_device target. RCCL sets the
# device-relevant include dirs on the rccl target in src/CMakeLists.txt (target_include_directories
# for ${HIPIFY_DIR}/src/device, .../device/network/unpack, ${GEN_DIR}, etc.), and -- on the ON path
# -- DeviceLinker.cmake's rccl_device_defs INTERFACE library re-derives them from the rccl target.
# Not duplicated here; see projects/rccl/src/CMakeLists.txt include-directories block and
# cmake/DeviceLinker.cmake L70-130.

# Add device sources to parent scope
set(DEVICE_SOURCES ${DEVICE_SOURCES} PARENT_SCOPE)
set(DEVICE_GEN_SOURCES ${DEVICE_GEN_SOURCES} PARENT_SCOPE)
set(DEVICE_REMOVED_SOURCES ${DEVICE_REMOVED_SOURCES} PARENT_SCOPE)
