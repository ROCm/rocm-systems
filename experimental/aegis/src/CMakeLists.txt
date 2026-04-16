# AegisBit core library

# Find HIP
find_package(hip QUIET CONFIG
  PATHS
    "${ROCM_PATH}/lib/cmake/hip"
    "/opt/rocm/lib/cmake/hip"
  NO_DEFAULT_PATH
)

# Find HSA runtime
find_library(HSA_RUNTIME_LIBRARY
  NAMES hsa-runtime64
  PATHS
    "${ROCM_PATH}/lib"
    "/opt/rocm/lib"
  NO_DEFAULT_PATH
)

# Collect source files
set(AEGISBIT_SOURCES
  disasm/Disassembler.cpp
  disasm/InstructionBuilder.cpp
  analysis/CFGBuilder.cpp
  analysis/CoalescingAnalyzer.cpp
  analysis/ScratchRegisters.cpp
  analysis/SourceMapper.cpp
  codegen/PayloadCompiler.cpp
  transform/ISAEncoder.cpp
  transform/IslandAllocator.cpp
  transform/JumpHeuristics.cpp
  transform/TrampolineEmitter.cpp
  transform/RelayEmitter.cpp
  transform/BridgeHelpers.cpp
  transform/SharedBodyStrategy.cpp
  transform/SwapPCSharedBodyStrategy.cpp
  transform/AdaptiveStrategy.cpp
  transform/TrampolineBridge.cpp
  analysis/SiteAnalyzer.cpp
  fixup/DescriptorUpdater.cpp
  codeobj/CodeObjectParser.cpp
  codeobj/NoteMetadataHandler.cpp
  codeobj/CodeObjectHandler.cpp
  runtime/HSAPoolManager.cpp
  runtime/InstrumentationPlanner.cpp
  runtime/KernargPool.cpp
  runtime/LoadedKernelCache.cpp
  runtime/PersistentBufferCache.cpp
  runtime/ProfilingResultsSink.cpp
  runtime/SignalMonitor.cpp
  runtime/TraceBufferAllocator.cpp
  runtime/RuntimeConfig.cpp
  runtime/KernelPatcher.cpp
)

# GPU-dependent sources (only build if HIP and HSA are available)
if(hip_FOUND AND HSA_RUNTIME_LIBRARY)
  message(STATUS "HIP and HSA runtime found, building GPU support")
  list(APPEND AEGISBIT_SOURCES
    intercept/DispatchInterceptor.cpp
    intercept/HSAInterceptor.cpp
    launch/KernelLauncher.cpp
    runtime/TracingEngine.cpp
  )
  set(AEGISBIT_HAS_GPU ON CACHE BOOL "GPU support available" FORCE)
else()
  message(STATUS "HIP or HSA runtime not found, building without GPU support")
  set(AEGISBIT_HAS_GPU OFF CACHE BOOL "GPU support available" FORCE)
endif()

# Create library
add_library(aegisbit STATIC ${AEGISBIT_SOURCES})

# Link LLVM components
llvm_map_components_to_libnames(llvm_libs ${LLVM_LINK_COMPONENTS})
target_link_libraries(aegisbit PUBLIC ${llvm_libs})

# Link HIP and HSA runtime if available
if(AEGISBIT_HAS_GPU)
  # Use hip::host for host-side HIP runtime (not hip::device which sets HIP language)
  target_link_libraries(aegisbit PUBLIC
    hip::host
  )
  # Also link HSA runtime for KernelLauncher
  if(HSA_RUNTIME_LIBRARY)
    target_link_libraries(aegisbit PUBLIC ${HSA_RUNTIME_LIBRARY})
  endif()
  target_compile_definitions(aegisbit PUBLIC AEGISBIT_HAS_GPU=1)
endif()

# Include directories
target_include_directories(aegisbit PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

# Compiler definitions
target_compile_definitions(aegisbit PUBLIC ${LLVM_DEFINITIONS})

# Shared library for LD_PRELOAD usage
if(AEGISBIT_HAS_GPU)
  add_library(aegisbit_shared SHARED ${AEGISBIT_SOURCES})
  target_link_libraries(aegisbit_shared PUBLIC ${llvm_libs} hip::host)
  if(HSA_RUNTIME_LIBRARY)
    target_link_libraries(aegisbit_shared PUBLIC ${HSA_RUNTIME_LIBRARY})
  endif()
  target_include_directories(aegisbit_shared PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
  )
  target_compile_definitions(aegisbit_shared PUBLIC ${LLVM_DEFINITIONS} AEGISBIT_HAS_GPU=1)
  set_target_properties(aegisbit_shared PROPERTIES OUTPUT_NAME aegisbit)

  # --- Portable binary: statically link utility libraries ---
  option(AEGISBIT_PORTABLE "Build a portable binary with static utility libs" OFF)
  if(AEGISBIT_PORTABLE)
    # Find static versions of utility libraries that LLVM pulls in.
    # This eliminates dependencies on distro-specific .so versions.
    foreach(_lib z zstd lzma bz2 elf dw numa)
      find_library(STATIC_${_lib} NAMES lib${_lib}.a
        PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib
        NO_DEFAULT_PATH)
      if(STATIC_${_lib})
        message(STATUS "Portable: static ${_lib} -> ${STATIC_${_lib}}")
        target_link_libraries(aegisbit_shared PRIVATE ${STATIC_${_lib}})
      else()
        message(WARNING "Portable: lib${_lib}.a not found, falling back to dynamic")
        target_link_libraries(aegisbit_shared PRIVATE ${_lib})
      endif()
    endforeach()

    # Version script: export only the HSA intercept symbols.
    # Hide LLVM internals to avoid conflicts with other LLVM-using libs
    # in the same process (e.g. Mesa, rocBLAS).
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/aegisbit.map"
"AEGISBIT_0.1 {
  global:
    hsa_*;
    aegisbit_*;
  local:
    *;
};
")
    target_link_options(aegisbit_shared PRIVATE
      "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/aegisbit.map"
      "-Wl,--as-needed"
      "-Wl,--exclude-libs,ALL"
    )

    # Note: libz.so remains a dynamic dep (pulled by LLVM cmake config).
    # This is fine — zlib is part of every Linux base system.

    # RPATH: look next to the .so first, then standard ROCm paths
    set_target_properties(aegisbit_shared PROPERTIES
      INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:/opt/rocm/lib"
      BUILD_RPATH   "$ORIGIN:/opt/rocm/lib"
    )
  endif()

  install(TARGETS aegisbit_shared LIBRARY DESTINATION lib)
endif()

install(PROGRAMS ${CMAKE_SOURCE_DIR}/tools/aegisbit DESTINATION tools)

# --- dist target: self-contained tarball ---
if(AEGISBIT_HAS_GPU)
  set(DIST_NAME "aegisbit-${CMAKE_PROJECT_VERSION}-linux-x86_64")
  configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/INSTALL.md.in"
    "${CMAKE_CURRENT_BINARY_DIR}/INSTALL.md"
    @ONLY
  )
  add_custom_target(dist
    DEPENDS aegisbit_shared
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${CMAKE_BINARY_DIR}/${DIST_NAME}"
    COMMAND ${CMAKE_COMMAND} -E make_directory   "${CMAKE_BINARY_DIR}/${DIST_NAME}/lib"
    COMMAND ${CMAKE_COMMAND} -E make_directory   "${CMAKE_BINARY_DIR}/${DIST_NAME}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy
      "$<TARGET_FILE:aegisbit_shared>"
      "${CMAKE_BINARY_DIR}/${DIST_NAME}/lib/libaegisbit.so"
    COMMAND strip --strip-unneeded "${CMAKE_BINARY_DIR}/${DIST_NAME}/lib/libaegisbit.so"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_SOURCE_DIR}/tools/aegisbit"
      "${CMAKE_BINARY_DIR}/${DIST_NAME}/bin/aegisbit"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_SOURCE_DIR}/README.md"
      "${CMAKE_BINARY_DIR}/${DIST_NAME}/README.md"
    COMMAND ${CMAKE_COMMAND} -E copy
      "${CMAKE_CURRENT_BINARY_DIR}/INSTALL.md"
      "${CMAKE_BINARY_DIR}/${DIST_NAME}/INSTALL.md"
    COMMAND tar czf "${CMAKE_BINARY_DIR}/${DIST_NAME}.tar.gz"
      -C "${CMAKE_BINARY_DIR}" "${DIST_NAME}"
    COMMENT "Packaging portable distribution -> ${DIST_NAME}.tar.gz"
    VERBATIM
  )
endif()
