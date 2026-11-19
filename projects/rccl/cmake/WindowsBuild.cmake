# Native Windows bring-up. Linux-only ROCr, SMI, RDMA, RAS and profiling
# dependencies are intentionally not required by this HIP/Win32 build.
find_package(hip REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GPU_TARGETS gfx1151 CACHE STRING "AMD GPU architectures")
set(ONLY_FUNCS "" CACHE STRING "Optional development collective filter")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/rccl")
configure_file(src/nccl.h.in "${PROJECT_BINARY_DIR}/include/nccl.h")
configure_file(src/nccl.h.in "${PROJECT_BINARY_DIR}/include/rccl/rccl.h")
set(RCCL_WINDOWS_COLLECT_SOURCES ON)
add_subdirectory(src)
unset(RCCL_WINDOWS_COLLECT_SOURCES)

# Retain the shared source list, selecting the existing Win32 implementations.
list(FILTER SRC_FILES INCLUDE REGEX "\\.(cc|cu|c)$")
list(FILTER SRC_FILES EXCLUDE REGEX "src/(os/linux|ras/|plugin/|gin/|rma/|diagnostics/|misc/(ibv|mlx5|ionic|alt_rsmi|gdr_|kernel_config|rocmwrap)|transport/net_ib|nccl_device/gin_)")
list(FILTER SRC_FILES EXCLUDE REGEX "src/(diagnostics\\.cc|misc/api_trace\\.c|misc/signals\\.cc|transport/net_telemetry\\.cc)$")
list(APPEND SRC_FILES src/plugin/net.cc)
file(GLOB WINDOWS_ND_SOURCES RELATIVE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/transport/net_nd/*.cc")
list(APPEND SRC_FILES ${WINDOWS_ND_SOURCES})
file(GLOB WINDOWS_RMA_SOURCES RELATIVE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/rma/*.cc")
list(APPEND SRC_FILES ${WINDOWS_RMA_SOURCES})
file(GLOB WINDOWS_OS_SOURCES RELATIVE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/os/windows*.cc")
list(REMOVE_ITEM WINDOWS_OS_SOURCES src/os/windows_enhcompat.cc)
list(APPEND SRC_FILES ${WINDOWS_OS_SOURCES} src/os/windows_rocmwrap.cc)

set(HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/scripts/stage_windows.py"
  "${PROJECT_SOURCE_DIR}" "${HIPIFY_DIR}" "${ROCM_PATH}"
  COMMAND_ERROR_IS_FATAL ANY)
set(GEN_DIR "${HIPIFY_DIR}/gensrc")
file(SHA256 "${PROJECT_SOURCE_DIR}/src/device/generate.py" _generator_hash)
file(SHA256 "${PROJECT_SOURCE_DIR}/src/device/ce_reduce/generate.py" _ce_generator_hash)
set(_generation_signature "${_generator_hash};${_ce_generator_hash};${ONLY_FUNCS}")
set(_generation_stamp "${PROJECT_BINARY_DIR}/windows-generation.txt")
set(_previous_generation "")
if(EXISTS "${_generation_stamp}")
  file(READ "${_generation_stamp}" _previous_generation)
endif()
if(NOT _previous_generation STREQUAL _generation_signature)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/src/device/generate.py"
  "${GEN_DIR}" OFF OFF OFF OFF OFF "${ONLY_FUNCS}"
  COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/src/device/ce_reduce/generate.py"
  "${GEN_DIR}/ce_reduce" COMMAND_ERROR_IS_FATAL ANY)
file(WRITE "${_generation_stamp}" "${_generation_signature}")
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tuner/embed_tuner_configs.py"
  "${PROJECT_SOURCE_DIR}/tuner" "${PROJECT_BINARY_DIR}/include/rccl_tuner_embedded_configs.h"
  COMMAND_ERROR_IS_FATAL ANY)

list(TRANSFORM SRC_FILES PREPEND "${HIPIFY_DIR}/")
file(GLOB_RECURSE GENERATED_SOURCES "${GEN_DIR}/*.cpp")
list(FILTER GENERATED_SOURCES EXCLUDE REGEX "/specialized/")
file(WRITE "${PROJECT_BINARY_DIR}/git_version.cpp" "const char* rcclGitHash = \"windows-bringup\";\n")
set_source_files_properties(${SRC_FILES} ${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)
add_library(rccl SHARED ${SRC_FILES} ${GENERATED_SOURCES} "${PROJECT_BINARY_DIR}/git_version.cpp")
set_target_properties(rccl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(rccl PRIVATE
  "${PROJECT_BINARY_DIR}/include" "${HIPIFY_DIR}/src" "${HIPIFY_DIR}/src/include"
  "${HIPIFY_DIR}/src/include/plugin" "${HIPIFY_DIR}/src/device"
  "${HIPIFY_DIR}/src/device/network/unpack" "${HIPIFY_DIR}/src/include/nccl_device"
  "${HIPIFY_DIR}/src/transport" "${HIPIFY_DIR}/src/devcomm" "${GEN_DIR}"
  "${PROJECT_SOURCE_DIR}/external/fmt/include")
target_compile_definitions(rccl PRIVATE NCCL_BUILD_SHARED_LIB __HIP_PLATFORM_AMD__
  NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS FMT_HEADER_ONLY
  HIP_UNCACHED_MEMORY HIP_FABRIC_API ROCM_VERSION=100200)
target_compile_options(rccl PRIVATE -x hip -fgpu-rdc -Wno-ignored-attributes
  -Wno-unused-command-line-argument -Wno-microsoft-cast)
foreach(arch IN LISTS GPU_TARGETS)
  target_compile_options(rccl PRIVATE "--offload-arch=${arch}")
  target_link_options(rccl PRIVATE "--offload-arch=${arch}")
endforeach()
target_link_options(rccl PRIVATE --hip-link -fgpu-rdc "LINKER:/errorlimit:0")
target_link_libraries(rccl PRIVATE hip::host "${ROCM_PATH}/lib/rocm-core.lib" ws2_32 iphlpapi advapi32 setupapi cfgmgr32)
install(TARGETS rccl RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
install(DIRECTORY "${PROJECT_BINARY_DIR}/include/" DESTINATION include)
