# RCCL target configuration (compile definitions, compile options, link
# libraries, link options, NPKit, rocprofiler-register, soversion).
#
# This module is an RCCL delta with no NCCL CMake counterpart. It is kept out of
# src/CMakeLists.txt so the NCCL-aligned source-collection and target-creation
# regions there stay small and easy to merge from upstream.
#
# IMPORTANT: the assembly-extract device-linker pipeline (cmake/DeviceLinker.cmake)
# is intentionally NOT included from here. It calls enable_language(RCCLDEV),
# which CMake only permits at file/directory scope (never inside a function() or
# block()). src/CMakeLists.txt therefore includes DeviceLinker.cmake at directory
# scope immediately after calling rccl_configure_target(); by then this function
# has already populated the rccl target's COMPILE_DEFINITIONS / INCLUDE_DIRECTORIES,
# which DeviceLinker.cmake reads back via get_target_property().

include(${CMAKE_CURRENT_LIST_DIR}/Npkit.cmake)

# Apply all RCCL-specific configuration to the given library target.
function(rccl_configure_target TARGET)
  ## Set RCCL compile definitions
  if(COLLTRACE)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_COLLTRACE)
  endif()
  if(ENABLE_MSCCL_KERNEL)
    message(WARNING "MSCCL is deprecated and will be removed in a future version of RCCL.")
    target_compile_definitions(${TARGET} PRIVATE COMPILE_MSCCL_KERNEL)
  endif()
  if(ENABLE_MSCCLPP)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_MSCCLPP)
  endif()

  if(NCCL_OS_LINUX)
    target_compile_definitions(${TARGET} PRIVATE NCCL_OS_LINUX)
  elseif(NCCL_OS_WINDOWS)
    target_compile_definitions(${TARGET} PRIVATE NCCL_OS_WINDOWS)
  endif()
  if(USE_AMDSMI)
    target_compile_definitions(${TARGET} PRIVATE USE_AMDSMI)
    if(AMDSMI_FABRIC_API)
      target_compile_definitions(${TARGET} PRIVATE AMDSMI_FABRIC_DIRECT)
    endif()
  else()
    if(HAVE_ROCM_SMI64CONFIG)
      target_compile_definitions(${TARGET} PRIVATE USE_ROCM_SMI64CONFIG)
    endif()
    if(HAVE_ROCM_SMI_THREAD_ONLY_MUTEX)
      target_compile_definitions(${TARGET} PRIVATE USE_ROCM_SMI_THREAD_ONLY_MUTEX)
    endif()
  endif()
  if(ENABLE_WARP_SPEED)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_WARP_SPEED)
  endif()
  if(ENABLE_ROCSHMEM)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_ROCSHMEM)
  endif()
  if("${hip_version_string}" VERSION_GREATER_EQUAL "7.12.60540")
    target_compile_definitions(${TARGET} PRIVATE HIP_VMM_UNCACHED_MEMORY)
  endif()

  # ==== rocSHMEM integration (optional) ====
  if (ENABLE_ROCSHMEM)
    add_rocshmem_targets()
    # Ensure rocSHMEM is fully built/installed before compiling rccl
    if (TARGET rocshmem_ext)
      add_dependencies(${TARGET} rocshmem_ext)
    endif()

    if (ROCSHMEM_INCLUDE_DIR)
      target_include_directories(${TARGET} PRIVATE ${ROCSHMEM_INCLUDE_DIR})
    endif()

    # Moved to where MSCCL target_links
    ## target_link_libraries(rccl PRIVATE ${ROCSHMEM_LIBRARY})
    target_link_libraries(${TARGET} PRIVATE ${IBVERBS})
  endif()

  # NPKit flags (RCCL-only; see cmake/Npkit.cmake)
  rccl_configure_npkit(${TARGET})

  if(PROFILE)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_PROFILING)
  endif()
  if(ROCTX_ENABLE)
    target_compile_definitions(${TARGET} PRIVATE ROCTX_ENABLE)
  else()
    target_compile_definitions(${TARGET} PRIVATE NVTX_NO_IMPL)
    target_compile_definitions(${TARGET} PRIVATE NVTX_DISABLE)
  endif()
  if(TRACE)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_TRACE)
  endif()
  if(QP_TRACKING)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_QP_TRACKING)
  endif()
  if(${HIP_CONTIGUOUS_MEMORY})
    target_compile_definitions(${TARGET} PRIVATE HIP_CONTIGUOUS_MEMORY)
    message(STATUS "HIP_CONTIGUOUS_MEMORY enabled")
  else()
    message(STATUS "HIP_CONTIGUOUS_MEMORY disabled")
  endif()
  if("${hip_version_string}" VERSION_GREATER_EQUAL "5.7.31920")
    target_compile_definitions(${TARGET} PRIVATE HIP_UNCACHED_MEMORY)
    message(STATUS "HIP_UNCACHED_MEMORY enabled")
  else()
    message(STATUS "HIP_UNCACHED_MEMORY disabled - requires HIP version >= 5.7.31920")
    # keep --hipcc-func-supp on older HIP and compiler
    if(NOT IFC_ENABLED)
      target_compile_options(${TARGET} PRIVATE --hipcc-func-supp)
      message(STATUS "--hipcc-func-supp enabled")
    else()
      message(STATUS "--hipcc-func-supp disabled")
    endif()
  endif()
  if (HIP_HOST_UNCACHED_MEMORY)
    target_compile_definitions(${TARGET} PRIVATE HIP_HOST_UNCACHED_MEMORY)
    message(STATUS "HIP_HOST_UNCACHED_MEMORY enabled")
  else()
    message(STATUS "HIP_HOST_UNCACHED_MEMORY disabled")
  endif()
  if(HIP_FABRIC_API_FUNC AND HIP_FABRIC_HANDLE_TYPE)
    # HIP_FABRIC_API is used as both a cmake variable (ON/OFF, for status messages below)
    # and a compile definition (bare #define, for #ifdef guards in C++). The names are
    # intentionally identical; the cmake variable is not queried by any downstream cmake logic.
    set(HIP_FABRIC_API ON)
    target_compile_definitions(${TARGET} PRIVATE HIP_FABRIC_API)
    message(STATUS "HIP Fabric API enabled (hipMemImportFromShareableHandle and hipMemFabricHandle_t found)")
  else()
    set(HIP_FABRIC_API OFF)
    if(NOT HIP_FABRIC_API_FUNC)
      message(STATUS "hipMemImportFromShareableHandle not found - disabling HIP_FABRIC_API")
    endif()
    if(NOT HIP_FABRIC_HANDLE_TYPE)
      message(STATUS "hipMemFabricHandle_t type not found - disabling HIP_FABRIC_API")
    endif()
  endif()
  if (BUILD_BFD)
    if (HAVE_BFD)
      target_compile_definitions(${TARGET} PRIVATE HAVE_BFD)
    endif()
    if (HAVE_DECL_BFD_GET_SECTION_FLAGS)
      target_compile_definitions(${TARGET} PRIVATE HAVE_DECL_BFD_GET_SECTION_FLAGS)
    endif()
    if (HAVE_DECL_BFD_GET_SECTION_VMA)
      target_compile_definitions(${TARGET} PRIVATE HAVE_DECL_BFD_GET_SECTION_VMA)
    endif()
    if (HAVE_TWO_ARG_BFD_SECTION_SIZE)
      target_compile_definitions(${TARGET} PRIVATE HAVE_TWO_ARG_BFD_SECTION_SIZE)
    endif()
  endif()
  if (IFC_ENABLED)
    target_compile_definitions(${TARGET} PRIVATE USE_INDIRECT_FUNCTION_CALL)
  endif()
  if(DEMANGLE_DIR)
    target_compile_definitions(${TARGET} PRIVATE "HAVE_CPLUS_DEMANGLE=1")
    target_compile_definitions(${TARGET} PRIVATE "HAVE_DECL_BASENAME=1")
  endif()
  if(LL128_ENABLED)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_LL128)
  endif()
  if(CE_ENABLED)
    target_compile_definitions(${TARGET} PRIVATE CE_BATCH_ASYNC_SUPPORTED)
    message(STATUS "CE_BATCH_ASYNC_SUPPORTED compile definition enabled")
  endif()

  ## Set RCCL compile options
  if (HAVE_PARALLEL_JOBS)
    target_compile_options(${TARGET} PRIVATE -parallel-jobs=12)
  endif()

  if (ROCM_VERSION VERSION_GREATER_EQUAL "60200")
    target_compile_options(${TARGET} PRIVATE --offload-compress)    # Compress GPU code at compile time.
    target_link_libraries(${TARGET} PRIVATE --offload-compress)     # Compress GPU code at link time.
    message(STATUS "--offload-compress enabled - ROCm version >= 6.2.0")
  else()
    message(STATUS "--offload-compress disabled - ROCm version < 6.2.0")
  endif()

  target_compile_options(${TARGET} PRIVATE -Werror=uninitialized)
  target_compile_options(${TARGET} PRIVATE -Werror=sometimes-uninitialized)
  target_compile_options(${TARGET} PRIVATE -Wall)
  target_compile_options(${TARGET} PRIVATE -Werror=deprecated-copy-with-user-provided-copy)
  target_compile_options(${TARGET} PRIVATE -Wno-format-nonliteral)
  target_compile_options(${TARGET} PRIVATE -Wno-unused-function)
  if(ENABLE_DEVICE_LINKER)
    target_compile_definitions(${TARGET} PRIVATE RCCL_DEVICE_LINKER)
    target_compile_options(${TARGET} PRIVATE --offload-host-only)
  else()
    target_compile_options(${TARGET} PRIVATE -fgpu-rdc)
  endif()

  if(QUIET_WARNINGS)
    target_compile_options(${TARGET} PRIVATE -Wno-invalid-offsetof)
    target_compile_options(${TARGET} PRIVATE -Wno-unused-result)
    target_compile_options(${TARGET} PRIVATE -Wno-macro-redefined)
    target_compile_options(${TARGET} PRIVATE -Wno-unused-label)
    target_compile_options(${TARGET} PRIVATE -Wno-unused-variable)
    target_compile_options(${TARGET} PRIVATE -Wno-unused-private-field)
    target_compile_options(${TARGET} PRIVATE -Wno-null-conversion)
    target_compile_options(${TARGET} PRIVATE -Wno-missing-braces)
  endif()

  ## Set RCCL compile and linker options for unit tests and code coverage
  if(ENABLE_CODE_COVERAGE)
    if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
      message(FATAL_ERROR "Code coverage is enabled, but the build type is '${CMAKE_BUILD_TYPE}'. "
                          "Code coverage requires 'Debug' build types to expose internal symbols. "
                          "Please set CMAKE_BUILD_TYPE to 'Debug' and reconfigure.")
    endif()

    message(STATUS "Code coverage is enabled with build type '${CMAKE_BUILD_TYPE}'.")

    target_compile_options(${TARGET} PRIVATE
      -fvisibility=default
      "SHELL:-Xarch_host -fprofile-instr-generate"
      "SHELL:-Xarch_host -fcoverage-mapping")

    set(COVERAGE_SHARED_LINKER_FLAGS
      -fprofile-generate
      -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN
    )

    set(COVERAGE_EXE_LINKER_FLAGS
      -fprofile-generate
      -Wl,--enable-new-dtags,--build-id=sha1,--rpath,$ORIGIN/../lib
    )

    target_link_options(${TARGET} PRIVATE ${COVERAGE_SHARED_LINKER_FLAGS})
    target_link_options(${TARGET} PRIVATE ${COVERAGE_EXE_LINKER_FLAGS})
  elseif(BUILD_TESTS) # Enable default/hidden visibility based on build type and ROCM_VERSION
    if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
      target_compile_options(${TARGET} PRIVATE -fvisibility=default)
    else()
      target_compile_options(${TARGET} PRIVATE -fvisibility=hidden)
    endif()
  else()  # Enable hidden visibility for library without tests/code coverage enabled
    target_compile_options(${TARGET} PRIVATE -fvisibility=hidden)
  endif()

  if (HAVE_KERNARG_PRELOAD)
    target_compile_options(${TARGET} PRIVATE -mllvm --amdgpu-kernarg-preload-count=16)
  endif()

  if (REPORT_KERNEL_RESOURCE_USE)
    target_link_options(${TARGET} PRIVATE -Rpass-analysis=kernel-resource-usage)
  endif()

  if (DUMP_ASM) # Save temporary files from kernel compilation
    message(STATUS "Disassembling librccl.so to asm")
    # Maintain symbols but without changing code.  Keep additional data in dwarf section of binary.
    target_compile_options(${TARGET} PRIVATE -gline-tables-only)
    set(OBJ_DUMP ${ROCM_PATH}/llvm/bin/llvm-objdump)

    add_custom_command(TARGET ${TARGET} POST_BUILD
      COMMENT "Disassembling RCCL library"
      COMMAND /bin/bash -c "${OBJ_DUMP} --offload-fatbin librccl.so"
      VERBATIM
    )
    foreach(GPUARCH ${GPU_TARGETS})
      add_custom_command(TARGET ${TARGET} POST_BUILD
          COMMENT "Disassembling RCCL library to dump assembly for ${GPUARCH}"
          COMMAND /bin/bash -c "${OBJ_DUMP} -d -l --source --symbolize-operands librccl.so.0.hipv4-amdgcn-amd-amdhsa--${GPUARCH} > librccl.${GPUARCH}.s"
          VERBATIM
      )
    endforeach()
  endif()

  ## NOTE: This is currently being handled by rocm-cmake, however may need to be re-enabled in the future
  #foreach(target ${GPU_TARGETS})
  #  target_compile_options(rccl PRIVATE --offload-arch=${target})
  #endforeach()

  if(BUILD_ADDRESS_SANITIZER)
    target_compile_options(${TARGET} PRIVATE -fsanitize=address)
    target_link_options(${TARGET} PRIVATE -fsanitize=address -shared-libasan)
    if(DEFINED ASAN_RUNTIME_DIR)
      target_link_options(${TARGET} PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
    endif()
  endif()
  if(TIMETRACE)
    target_compile_options(${TARGET} PRIVATE -ftime-trace)
  endif()
  if (FAULT_INJECTION)
    target_compile_definitions(${TARGET} PRIVATE ENABLE_FAULT_INJECTION)
    message(STATUS "Fault injection enabled")
  endif()

  if (NOT DWORDX4_INTRINSICS)
    message(STATUS "Force disable dwordx4 intrinsics")
    target_compile_definitions(${TARGET} PRIVATE DWORDX4_INTRINSICS_FORCE_OFF)
  endif()
  ## Set RCCL linked library directories
  target_link_directories(${TARGET} PRIVATE ${SMI_LIB_DIR})

  if (ROCM_VERSION VERSION_GREATER_EQUAL "60100")
      option(RCCL_ROCPROFILER_REGISTER "Enable rocprofiler-register support" ON)
  else()
      if(RCCL_ROCPROFILER_REGISTER)
          message(AUTHOR_WARNING "RCCL_ROCPROFILER_REGISTER is not valid option for ROCm < 6.2. Current ROCm version: ${ROCM_VERSION}")
      endif()
      set(RCCL_ROCPROFILER_REGISTER OFF CACHE BOOL "" FORCE)
  endif()
  if(RCCL_ROCPROFILER_REGISTER)
    find_package(rocprofiler-register REQUIRED)
    target_compile_definitions(${TARGET} PRIVATE RCCL_ROCPROFILER_REGISTER=1)
    target_link_libraries(
        ${TARGET} PRIVATE rocprofiler-register::rocprofiler-register)
  endif()

  ## Set RCCL linked libraries
  if (HAVE_BFD)
    target_link_libraries(${TARGET} PRIVATE bfd)
    if(HAVE_IBERTY)
      target_link_libraries(${TARGET} PRIVATE iberty z)
    endif()
  endif()
  if (ROCTX_ENABLE)
    target_link_libraries(${TARGET} PRIVATE ${ROCTX_LIB})
  endif()
  if(NOT ENABLE_DEVICE_LINKER)
    target_link_libraries(${TARGET} PRIVATE   -fgpu-rdc)             # Required when linking relocatable device code
  endif()
  target_link_libraries(${TARGET} PRIVATE   Threads::Threads)
  target_link_libraries(${TARGET} INTERFACE hip::host)
  if(ENABLE_DEVICE_LINKER)
    # Device code is compiled separately by the DeviceLinker pipeline.
    # Link hip::host for the runtime library but add -x hip manually so HIP
    # builtins (warpSize, etc.) are available. Crucially, do NOT propagate
    # --offload-arch from hip::device — that would make the compiler try to
    # compile device code in host-only .cc files.
    target_link_libraries(${TARGET} PRIVATE hip::host)
    target_compile_options(${TARGET} PRIVATE -x hip)
    # api_trace.c exists only to include api_trace.h under strict C rules; global
    # -x hip makes Clang treat it as C++ and hits #error there. Per-file -x c is
    # appended after target flags so this unit is compiled as C (same driver).
    set_source_files_properties(
      "${HIPIFY_DIR}/src/misc/api_trace.c"
      PROPERTIES
        COMPILE_OPTIONS "-x;c"
    )
  else()
    target_link_libraries(${TARGET} PRIVATE hip::device)
  endif()
  target_link_libraries(${TARGET} PRIVATE   dl)
  target_link_libraries(${TARGET} PRIVATE   ${SMI_LIBRARIES})
  # Wrap fmt in $<BUILD_INTERFACE:...> so it's only used while building rccl
  # and is omitted from the exported rccl-targets export set. Without this,
  # a FetchContent-built fmt-header-only target leaks into rccl's link
  # interface and breaks install(EXPORT) for static builds.
  target_link_libraries(${TARGET} PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)
  if(ENABLE_ROCSHMEM)
    target_link_libraries(${TARGET} PRIVATE ${ROCSHMEM_LIBRARY})
    target_link_libraries(${TARGET} PRIVATE ${IBVERBS})
    # librocshmem.a is compiled with -fgpu-rdc (relocatable device code).
    # The __hip_gpubin_handle_* symbols referenced by __hip_module_ctor in the
    # librocshmem.a objects are only defined by the RDC device-link step.
    # --hip-link tells amdclang++ to perform that step as part of the final
    # librccl.so link; -fgpu-rdc marks the link as RDC-mode.  --offload-arch
    # flags use RCCL's GPU_TARGETS (stripped of feature suffixes like :xnack-)
    # so the device-link covers every arch RCCL is built for.
    # RCCL's own device code (already embedded in device_build/common.o by the
    # custom device-linker pipeline) is NOT compiled with -fgpu-rdc and is
    # therefore unaffected by this device-link step.
    set(_rocshmem_offload_arch_flags "")
    foreach(_gpu_raw ${GPU_TARGETS})
      string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
      list(APPEND _rocshmem_offload_arch_flags "--offload-arch=${_gpu}")
    endforeach()
    list(REMOVE_DUPLICATES _rocshmem_offload_arch_flags)
    target_link_options(${TARGET} PRIVATE -fgpu-rdc --hip-link ${_rocshmem_offload_arch_flags})
  endif()

  ## Set RCCL link options
  ## Find out available memory
  execute_process(
    COMMAND bash "-c" "test -r /sys/fs/cgroup/memory.max && cat /sys/fs/cgroup/memory.max || echo"
    OUTPUT_VARIABLE memory_max_string)
  if (${memory_max_string} MATCHES "^[0-9]+")
    math(EXPR detected_memory_gb "${memory_max_string} / (1024 * 1024 * 1024)")
  else()
    execute_process(
      COMMAND bash "-c" "free | grep -o '[[:digit:]]*' | head -1"
      OUTPUT_VARIABLE memory_max_string)
    ## memory_max_string holds the free memory in KB
    if (${memory_max_string} MATCHES "^[0-9]+")
      math(EXPR detected_memory_gb "${memory_max_string} / (1024 * 1024)") ## KB to GB conversion
    else()
      cmake_host_system_information(RESULT memory_max_string QUERY AVAILABLE_PHYSICAL_MEMORY )
      math(EXPR detected_memory_gb "${memory_max_string} / 1024")
    endif()
  endif()
  set(RCCL_MAX_MEMORY ${detected_memory_gb} CACHE STRING "Maximum memory (in GB) during linking")
  set(RCCL_MEMORY_PER_LINK_JOB 32 CACHE STRING "Memory (in GB) reserved for each linker job")
  # Use the minimum of detected memory and user-specified memory
  if (RCCL_MAX_MEMORY LESS detected_memory_gb)
    set(memory_in_gb ${RCCL_MAX_MEMORY})
  else()
    set(memory_in_gb ${detected_memory_gb})
  endif()

  ## Reserve 16GB for each linker job. Limit max number of linker jobs to 16
  if (HAVE_PARALLEL_JOBS)
    math(EXPR num_linker_jobs "(${memory_in_gb} + ${RCCL_MEMORY_PER_LINK_JOB} - 1) / ${RCCL_MEMORY_PER_LINK_JOB}")
    if (${num_linker_jobs} GREATER_EQUAL "16")
      set(num_linker_jobs "16")
    elseif (${num_linker_jobs} LESS_EQUAL "0")
      set(num_linker_jobs "1")
    endif()
    message(STATUS "Use ${num_linker_jobs} jobs for linking")
    target_link_options(${TARGET} PRIVATE -parallel-jobs=${num_linker_jobs})       # Use multiple threads to link
  endif()
  if(BUILD_ADDRESS_SANITIZER)
    target_link_options(${TARGET} PRIVATE -fuse-ld=lld)
    if(NOT ENABLE_DEVICE_LINKER)
      ## ---- ASAN device-link acceleration ----------------------------------------
      ## With -fgpu-rdc, the offload linker merges device bitcode from all object
      ## files and performs full LTO (optimization + codegen) for each GPU target.
      ## --lto-O0 + --lto-CGO0: skip LTO optimization and backend optimization.
      ## --------------------------------------------------------------------------
      target_link_options(${TARGET} PRIVATE "SHELL:-Xoffload-linker --lto-O0")
      target_link_options(${TARGET} PRIVATE "SHELL:-Xoffload-linker --lto-CGO0")
    endif()
  endif()
  if(CMAKE_BUILD_SUBTYPE MATCHES "DebugFast" AND NOT ENABLE_DEVICE_LINKER)
    target_link_options(${TARGET} PRIVATE "SHELL:-Xoffload-linker --lto-O0")
    target_link_options(${TARGET} PRIVATE "SHELL:-Xoffload-linker --lto-CGO0")
  endif()
  if(TIMETRACE)
    target_link_options(${TARGET} PRIVATE -ftime-trace)
  endif()

  if(NOT BUILD_SHARED_LIBS)
    message(STATUS "Building static RCCL library")
  else()
    message(STATUS "Building shared RCCL library")
  endif()
  if (HAVE_KERNARG_PRELOAD AND NOT ENABLE_DEVICE_LINKER)
    target_link_options(${TARGET} PRIVATE "SHELL:-Xoffload-linker -mllvm=-amdgpu-kernarg-preload-count=16")
  endif()

  if(ENABLE_MSCCLPP)
    include(${RCCL_SOURCE_DIR}/cmake/MSCCLPP.cmake)
  endif()

  ## Track linking time
  set_property(TARGET ${TARGET} PROPERTY RULE_LAUNCH_LINK "${CMAKE_COMMAND} -E time")

  ## Setup librccl.so version
  rocm_set_soversion(${TARGET} "1.0")
endfunction()
