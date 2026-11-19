# Native HIP/Win32 build. GPU targets and SDK capabilities are supplied by the
# caller and selected packages rather than assumed from the development host.
if(CMAKE_VERSION VERSION_LESS 3.22)
  message(FATAL_ERROR "The native Windows RCCL build requires CMake 3.22 or newer.")
endif()
set(GPU_TARGETS "" CACHE STRING "AMD GPU targets (for example gfx1151;gfx1201)")
if(NOT GPU_TARGETS)
  message(FATAL_ERROR "Set GPU_TARGETS explicitly for the intended AMD GPUs, for example -DGPU_TARGETS=gfx1151. No machine-specific architecture is assumed.")
endif()

option(BUILD_SHARED_LIBS "Build shared RCCL libraries" ON)
if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "The native Windows path currently supports BUILD_SHARED_LIBS=ON only.")
endif()
option(RCCL_BUILD_SMOKE_TEST "Build the single-GPU Windows RCCL smoke test" OFF)
option(BUILD_ALL_UNROLLS "Generate all supported kernel unroll factors" OFF)
option(FAULT_INJECTION "Enable device fault-injection instrumentation" OFF)
option(TRACE "Enable additional tracing" OFF)
option(TIMETRACE "Emit compiler time traces" OFF)
option(ENABLE_COMPRESS "Compress HIP device code" OFF)
set(ONLY_FUNCS "" CACHE STRING "Optional development collective filter")
set(RCCL_STAGE_JOBS 8 CACHE STRING "Parallel HIP source-staging jobs")
if(NOT RCCL_STAGE_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "RCCL_STAGE_JOBS must be a positive integer.")
endif()
foreach(_unsupported BUILD_TESTS BUILD_ADDRESS_SANITIZER BUILD_BFD BUILD_LOCAL_GPU_TARGET_ONLY
    BUILD_NCCL4PY BUILD_PLUGIN_EXAMPLES BUILD_PROFILER_ACCL BUILD_PROFILER_INSPECTOR
    ENABLE_CODE_COVERAGE ENABLE_DEVICE_LINKER ENABLE_IFC ENABLE_NET_PROFILING
    ENABLE_RCCL_EP_TESTS ENABLE_RCCL_EP_IN_LIBRCCL ENABLE_ROCSHMEM ENABLE_ROCSHMEM_GIN
    ENABLE_TDM_SIMPLE ENABLE_TDM_PRIM_LL128 ENABLE_WARP_SPEED GENERATE_SYM_KERNELS
    INSTALL_DEPENDENCIES NCCL_EMIT_LLVM_IR NCCL_EMIT_LTO_IR ROCTX SQTT_ENABLED
    RCCL_ROCPROFILER_REGISTER)
  option(${_unsupported} "Not yet supported by the native Windows RCCL build" OFF)
  if(${_unsupported})
    message(FATAL_ERROR "${_unsupported}=ON is not implemented by the native Windows RCCL build. Set it to OFF; use RCCL_BUILD_SMOKE_TEST for the Windows smoke test and explicit GPU_TARGETS for architecture selection.")
  endif()
endforeach()
if(DEFINED ENABLE_FULL_COVERAGE AND NOT ENABLE_FULL_COVERAGE STREQUAL "OFF")
  message(FATAL_ERROR "ENABLE_FULL_COVERAGE is not implemented by the native Windows RCCL build; set it to OFF.")
endif()

find_package(hip REQUIRED CONFIG)
find_package(rocm-core REQUIRED CONFIG)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Version and dependencies belong to the selected package, not a fixed SDK.
set(_rocm_version_header "${ROCM_CORE_INCLUDE_DIR}/rocm-core/rocm_version.h")
if(NOT EXISTS "${_rocm_version_header}")
  message(FATAL_ERROR "The selected rocm-core package has no rocm-core/rocm_version.h version header.")
endif()
file(READ "${_rocm_version_header}" _rocm_version_text)
foreach(_component MAJOR MINOR PATCH)
  if(NOT _rocm_version_text MATCHES "#[ \t]*define[ \t]+ROCM_VERSION_${_component}[ \t]+([0-9]+)")
    message(FATAL_ERROR "Cannot read ROCM_VERSION_${_component} from ${_rocm_version_header}.")
  endif()
  set(ROCM_${_component}_VERSION "${CMAKE_MATCH_1}")
endforeach()
math(EXPR ROCM_VERSION "10000 * ${ROCM_MAJOR_VERSION} + 100 * ${ROCM_MINOR_VERSION} + ${ROCM_PATCH_VERSION}")
message(STATUS "Windows RCCL: ROCm ${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}, GPUs ${GPU_TARGETS}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_rocm_version_header}")

# These probe compile-time API/type availability, not runtime capabilities.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES hip::host)
check_cxx_source_compiles("#include <hip/hip_runtime_api.h>\nint main() { auto flag = hipDeviceMallocUncached; (void)flag; }" RCCL_WINDOWS_HIP_UNCACHED_MEMORY)
check_cxx_source_compiles("#include <hip/hip_runtime_api.h>\nint main() { hipMemFabricHandle_t handle{}; (void)handle; }" RCCL_WINDOWS_HIP_FABRIC_HANDLE)
cmake_pop_check_state()
# hipHostMallocUncached is declared in Windows headers but documented as Linux-only.
# Do not define HIP_HOST_UNCACHED_MEMORY from declaration availability: that makes
# every host allocation fail on Windows. Keep the gfx950 initialization safety
# check in place for devices that require this unsupported allocation mode.

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/include/rccl")
configure_file(src/nccl.h.in "${PROJECT_BINARY_DIR}/include/nccl.h")
configure_file(src/nccl.h.in "${PROJECT_BINARY_DIR}/include/rccl/rccl.h")
set(RCCL_WINDOWS_COLLECT_SOURCES ON)
add_subdirectory(src)
unset(RCCL_WINDOWS_COLLECT_SOURCES)

# Reuse the shared source list and select the existing Win32 implementations.
list(FILTER SRC_FILES INCLUDE REGEX "\\.(cc|cu|c|cpp)$")
list(FILTER SRC_FILES EXCLUDE REGEX "src/(os/linux|ras/|plugin/|gin/|rma/|diagnostics/|misc/(ibv|mlx5|ionic|alt_rsmi|gdr_|kernel_config|rocmwrap)|transport/net_ib|nccl_device/gin_)")
list(FILTER SRC_FILES EXCLUDE REGEX "src/(diagnostics\\.cc|misc/api_trace\\.c|misc/signals\\.cc|transport/net_telemetry\\.cc)$")
list(APPEND SRC_FILES src/plugin/net.cc)
file(GLOB WINDOWS_ND_SOURCES CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/transport/net_nd/*.cc")
file(GLOB WINDOWS_RMA_SOURCES CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/rma/*.cc")
file(GLOB WINDOWS_OS_SOURCES CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/src/os/windows*.cc")
list(REMOVE_ITEM WINDOWS_OS_SOURCES src/os/windows_enhcompat.cc)
list(APPEND SRC_FILES ${WINDOWS_ND_SOURCES} ${WINDOWS_RMA_SOURCES} ${WINDOWS_OS_SOURCES})
list(REMOVE_DUPLICATES SRC_FILES)

# Stage during the build so edits to original sources invalidate compilation.
set(HIPIFY_DIR "${PROJECT_BINARY_DIR}/hipify")
file(GLOB_RECURSE _stage_inputs CONFIGURE_DEPENDS RELATIVE "${PROJECT_SOURCE_DIR}"
  "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.hpp"
  "${PROJECT_SOURCE_DIR}/src/*.cuh" "${PROJECT_SOURCE_DIR}/src/*.cc"
  "${PROJECT_SOURCE_DIR}/src/*.cu" "${PROJECT_SOURCE_DIR}/src/*.c"
  "${PROJECT_SOURCE_DIR}/src/*.cpp")
set(_stage_outputs ${_stage_inputs})
list(TRANSFORM _stage_outputs PREPEND "${HIPIFY_DIR}/")
list(TRANSFORM _stage_inputs PREPEND "${PROJECT_SOURCE_DIR}/")
set(_stage_script "${PROJECT_SOURCE_DIR}/cmake/scripts/stage_windows.py")
set(_stage_stamp "${PROJECT_BINARY_DIR}/windows-stage.stamp")
set(_stage_depfile "${PROJECT_BINARY_DIR}/windows-stage.d")
set(_stage_options --jobs "${RCCL_STAGE_JOBS}")
foreach(_tool PERL BASH HIPIFY_PERL)
  set(RCCL_${_tool} "" CACHE FILEPATH "Optional ${_tool} executable/script for HIP staging")
  if(RCCL_${_tool})
    string(TOLOWER "${_tool}" _tool_option)
    string(REPLACE "hipify_perl" "hipify" _tool_option "${_tool_option}")
    list(APPEND _stage_options "--${_tool_option}" "${RCCL_${_tool}}")
  endif()
endforeach()
if(FAULT_INJECTION)
  list(APPEND _stage_options --fault-injection)
endif()
add_custom_command(OUTPUT "${_stage_stamp}"
  COMMAND "${Python3_EXECUTABLE}" "${_stage_script}" "${PROJECT_SOURCE_DIR}" "${HIPIFY_DIR}" "${ROCM_PATH}"
    --stamp "${_stage_stamp}" --depfile "${_stage_depfile}" --depfile-target "${_stage_stamp}" ${_stage_options}
  DEPENDS ${_stage_inputs} "${_stage_script}"
    "${PROJECT_SOURCE_DIR}/cmake/scripts/add_unroll.sh" "${PROJECT_SOURCE_DIR}/cmake/scripts/add_faults.sh"
  BYPRODUCTS ${_stage_outputs}
  DEPFILE "${_stage_depfile}"
  COMMENT "Staging RCCL HIP sources" VERBATIM)
add_custom_target(rccl_windows_stage DEPENDS "${_stage_stamp}")
set(NCCL_GIN_ROCSHMEM_GDA_AVAILABLE 0)
set(NCCL_GIN_ANVIL_SDMA_AVAILABLE 0)
configure_file("${PROJECT_SOURCE_DIR}/src/include/nccl_device/gin/rccl_gin_config.h.in"
  "${HIPIFY_DIR}/src/include/nccl_device/gin/rccl_gin_config.h" @ONLY)

set(GEN_DIR "${HIPIFY_DIR}/gensrc")
set(_generator_inputs "${PROJECT_SOURCE_DIR}/src/device/generate.py"
  "${PROJECT_SOURCE_DIR}/src/device/ce_reduce/generate.py"
  "${PROJECT_SOURCE_DIR}/src/device/ce_reduce/ce_reduce_impl.h.in"
  "${PROJECT_SOURCE_DIR}/src/device/ce_reduce/ce_reduce_launcher.cpp.in")
set(_generation_signature "${ONLY_FUNCS};${BUILD_ALL_UNROLLS}")
foreach(_input IN LISTS _generator_inputs)
  file(SHA256 "${_input}" _input_hash)
  string(APPEND _generation_signature ";${_input_hash}")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_generator_inputs})
set(_generation_stamp "${PROJECT_BINARY_DIR}/windows-generation.txt")
set(_previous_generation "")
if(EXISTS "${_generation_stamp}")
  file(READ "${_generation_stamp}" _previous_generation)
endif()
if(NOT _previous_generation STREQUAL _generation_signature OR NOT EXISTS "${GEN_DIR}/host_table.cpp"
    OR NOT EXISTS "${GEN_DIR}/ce_reduce/ce_reduce_impl.h")
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/src/device/generate.py"
    "${GEN_DIR}" OFF OFF OFF OFF "${BUILD_ALL_UNROLLS}" "${ONLY_FUNCS}" COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/src/device/ce_reduce/generate.py"
    "${GEN_DIR}/ce_reduce" COMMAND_ERROR_IS_FATAL ANY)
  file(WRITE "${_generation_stamp}" "${_generation_signature}")
endif()
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tuner/embed_tuner_configs.py"
  "${PROJECT_SOURCE_DIR}/tuner" "${PROJECT_BINARY_DIR}/include/rccl_tuner_embedded_configs.h" COMMAND_ERROR_IS_FATAL ANY)

find_package(Git QUIET)
set(_git_hash "Unknown")
if(Git_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" rev-parse --short=12 HEAD
    OUTPUT_VARIABLE _git_hash OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${PROJECT_SOURCE_DIR}" diff HEAD --quiet RESULT_VARIABLE _git_dirty)
  if(_git_dirty EQUAL 1)
    string(APPEND _git_hash "+")
  endif()
endif()
file(CONFIGURE OUTPUT "${PROJECT_BINARY_DIR}/git_version.cpp" CONTENT "const char* rcclGitHash = \"${_git_hash}\";\n")
list(TRANSFORM SRC_FILES PREPEND "${HIPIFY_DIR}/")
file(GLOB_RECURSE GENERATED_SOURCES "${GEN_DIR}/*.cpp")
list(FILTER GENERATED_SOURCES EXCLUDE REGEX "/specialized/")
set_source_files_properties(${SRC_FILES} PROPERTIES GENERATED TRUE LANGUAGE CXX)
set_source_files_properties(${GENERATED_SOURCES} PROPERTIES LANGUAGE CXX)
add_library(rccl SHARED ${SRC_FILES} ${GENERATED_SOURCES} "${PROJECT_BINARY_DIR}/git_version.cpp")
add_dependencies(rccl rccl_windows_stage)
set_target_properties(rccl PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
target_include_directories(rccl PUBLIC "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>" "$<INSTALL_INTERFACE:include>"
  PRIVATE "${HIPIFY_DIR}/src" "${HIPIFY_DIR}/src/include" "${HIPIFY_DIR}/src/include/plugin"
  "${HIPIFY_DIR}/src/device" "${HIPIFY_DIR}/src/device/network/unpack" "${HIPIFY_DIR}/src/include/nccl_device"
  "${HIPIFY_DIR}/src/transport" "${HIPIFY_DIR}/src/devcomm" "${GEN_DIR}" "${PROJECT_SOURCE_DIR}/external/fmt/include")
target_compile_definitions(rccl PRIVATE NCCL_BUILD_SHARED_LIB NOMINMAX WIN32_LEAN_AND_MEAN
  _CRT_SECURE_NO_WARNINGS FMT_HEADER_ONLY "ROCM_VERSION=${ROCM_VERSION}")
foreach(_capability HIP_UNCACHED_MEMORY HIP_FABRIC_HANDLE)
  if(RCCL_WINDOWS_${_capability})
    if(_capability STREQUAL "HIP_FABRIC_HANDLE")
      target_compile_definitions(rccl PRIVATE HIP_FABRIC_API)
    else()
      target_compile_definitions(rccl PRIVATE ${_capability})
    endif()
  endif()
endforeach()
if(FAULT_INJECTION)
  target_compile_definitions(rccl PRIVATE ENABLE_FAULT_INJECTION)
endif()
if(TRACE)
  target_compile_definitions(rccl PRIVATE ENABLE_TRACE)
endif()
if(TIMETRACE)
  target_compile_options(rccl PRIVATE -ftime-trace)
  target_link_options(rccl PRIVATE -ftime-trace)
endif()
if(ENABLE_COMPRESS)
  target_compile_options(rccl PRIVATE --offload-compress)
  target_link_options(rccl PRIVATE --offload-compress)
endif()
target_compile_options(rccl PRIVATE -x hip -fgpu-rdc -Wno-ignored-attributes
  -Wno-unused-command-line-argument -Wno-microsoft-cast)
foreach(_arch IN LISTS GPU_TARGETS)
  target_compile_options(rccl PRIVATE "--offload-arch=${_arch}")
  target_link_options(rccl PRIVATE "--offload-arch=${_arch}")
endforeach()
target_link_options(rccl PRIVATE --hip-link -fgpu-rdc)
target_link_libraries(rccl PUBLIC hip::host PRIVATE rocm-core ws2_32 iphlpapi advapi32 setupapi cfgmgr32)

if(RCCL_BUILD_SMOKE_TEST)
  enable_testing()
  add_executable(rccl_windows_smoke "${PROJECT_SOURCE_DIR}/test/windows/smoke.cpp")
  target_link_libraries(rccl_windows_smoke PRIVATE rccl hip::host)
  set_target_properties(rccl_windows_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test/windows")
  add_custom_command(TARGET rccl_windows_smoke POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different $<TARGET_RUNTIME_DLLS:rccl_windows_smoke> $<TARGET_FILE_DIR:rccl_windows_smoke>
    COMMAND_EXPAND_LISTS VERBATIM)
  add_test(NAME rccl_windows_smoke COMMAND rccl_windows_smoke)
  set_tests_properties(rccl_windows_smoke PROPERTIES SKIP_RETURN_CODE 77
    ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:hip::amdhip64>")
endif()
install(TARGETS rccl RUNTIME DESTINATION bin ARCHIVE DESTINATION lib)
install(FILES "${PROJECT_BINARY_DIR}/include/nccl.h" DESTINATION include)
install(FILES "${PROJECT_BINARY_DIR}/include/rccl/rccl.h" DESTINATION include/rccl)
