# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

cmake_minimum_required(VERSION 3.16)

if(BUILD_TESTS)

  option(OPENMP_TESTS_ENABLED "Enable OpenMP for unit tests" OFF)
  option(ENABLE_MPI_TESTS "Enable MPI-based tests" OFF)
  #Automatically enabled when MPI-based tests available
  option(RCCL_HAS_RMA_IB_PROXY "Build the IB Proxy RMA transport-level tests" ON)
  option(ENABLE_HOST_API_TESTS "Build the host-API tests" ON)


  message("Building rccl unit tests (Installed in /test/rccl-UnitTests)")
  if(ENABLE_MPI_TESTS)
    message("MPI-based tests are enabled")
  endif()

  # CPU-only generator unit test: guards the -fgpu-rdc device-function dispatch
  # codegen in src/device/generate.py. Needs no GPU and no built library.
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(
    NAME rccl-generate-device-table
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/src/device/test_generate_device_table.py"
  )

  # CPU-only generator unit test: guards src/device/ce_reduce/generate.py's
  # per-instantiation TU splitting (ce_reduce_impl.h.in / ce_reduce_launcher.cpp.in
  # template expansion). Needs no GPU and no built library.
  add_test(
    NAME rccl-generate-ce-reduce
    COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/src/device/ce_reduce/test_generate_ce_reduce.py"
  )

  # The kernel-count leak guards (src/device/test_generate_kernel_counts.py and
  # src/device/symmetric/test_generate_symmetric.py) are intentionally NOT
  # registered as ctest cases here: nothing in RCCL CI runs `ctest`, so an
  # add_test() would never gate. They run in CI via the CPU-only host-test
  # pipeline instead -- see test/host/run_host_tests.sh (the `guards` phase,
  # folded into `run`).

  # CPU-only net_telemetry unit test: net_telemetry.cc depends only on libc and
  # pthread, so this compiles it directly with its own harness (no GPU, no gtest,
  # no librccl) and runs under CTest, returning non-zero on any failed CHECK.
  find_package(Threads REQUIRED)
  add_executable(rccl-UnitTestsNetTelemetry
    ${PROJECT_SOURCE_DIR}/test/net_telemetry_test.cpp
    ${PROJECT_SOURCE_DIR}/src/transport/net_telemetry.cc
  )
  target_include_directories(rccl-UnitTestsNetTelemetry PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)
  target_link_libraries(rccl-UnitTestsNetTelemetry PRIVATE Threads::Threads)
  add_test(NAME rccl-UnitTestsNetTelemetry COMMAND rccl-UnitTestsNetTelemetry)
  rocm_install(TARGETS rccl-UnitTestsNetTelemetry COMPONENT tests)

  if (ENABLE_CODE_COVERAGE)
    add_compile_options("SHELL:-Xarch_host -fprofile-instr-generate" "SHELL:-Xarch_host -fcoverage-mapping")
    add_link_options("SHELL:-Xarch_host -fprofile-instr-generate" "SHELL:-Xarch_host -fcoverage-mapping")
    # Allow ProcessIsolatedTestRunner.cpp to call __llvm_profile_write_file()
    # explicitly before _exit() so child-process coverage data is flushed.
    add_compile_definitions(RCCL_TEST_CODE_COVERAGE=1)
  endif()

  find_package(hsa-runtime64 PATHS /opt/rocm )
  if(${hsa-runtime64_FOUND})
    message("hsa-runtime64 found @  ${hsa-runtime64_DIR} ")
  else()
    message("find_package did NOT find hsa-runtime64, finding it the OLD Way")
    message("Looking for header files in ${ROCR_INC_DIR}")
    message("Looking for library files in ${ROCR_LIB_DIR}")

    # Search for ROCr header file in user defined locations
    find_path(ROCR_HDR hsa/hsa.h PATHS ${ROCR_INC_DIR} "/opt/rocm" PATH_SUFFIXES include REQUIRED)
    include_directories(${ROCR_HDR})

    # Search for ROCr library file in user defined locations
    find_library(ROCR_LIB ${CORE_RUNTIME_TARGET} PATHS ${ROCR_LIB_DIR} "/opt/rocm" PATH_SUFFIXES lib lib64 REQUIRED)
  endif()

  if(OPENMP_TESTS_ENABLED)
    find_package(OpenMP REQUIRED)
  endif()

  # MPI configuration
  if(ENABLE_MPI_TESTS)
    # Set MPI path: 1) environment variable (always wins), 2) CMake variable, 3) default
    if(DEFINED ENV{MPI_PATH})
      set(MPI_PATH "$ENV{MPI_PATH}" CACHE PATH "Path to MPI installation" FORCE)
    elseif(NOT DEFINED MPI_PATH)
      set(MPI_PATH "/opt/ompi" CACHE PATH "Path to MPI installation")
    endif()

    # Verify MPI path exists
    if(NOT EXISTS ${MPI_PATH})
        message(WARNING "MPI_PATH does not exist: ${MPI_PATH}")
        message(WARNING "Please set MPI_PATH to your MPI installation directory")
        message(FATAL_ERROR "MPI installation not found")
    endif()

    message(STATUS "Using MPI installation at: ${MPI_PATH}")

    # Find required MPI library
    find_library(MPI_LIBRARY
        NAMES mpi
        PATHS ${MPI_PATH}/lib ${MPI_PATH}/lib64
        NO_DEFAULT_PATH
        REQUIRED
    )

    if(NOT MPI_LIBRARY)
        message(FATAL_ERROR "Could not find MPI library (libmpi.so) in ${MPI_PATH}/lib or ${MPI_PATH}/lib64")
    endif()

    # Set up MPI variables
    set(MPI_CXX_LIBRARIES ${MPI_LIBRARY})
    set(MPI_CXX_INCLUDE_DIRS ${MPI_PATH}/include)
    set(MPI_CXX_LINK_FLAGS "-L${MPI_PATH}/lib -Wl,-rpath,${MPI_PATH}/lib")
    set(MPIEXEC_EXECUTABLE ${MPI_PATH}/bin/mpirun CACHE FILEPATH "MPI executable")

    # Add link directories for MPI
    link_directories(${MPI_PATH}/lib)

    message(STATUS "MPI library: ${MPI_CXX_LIBRARIES}")
    message(STATUS "MPI include: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")
  endif()

  include_directories(${GTEST_INCLUDE_DIRS} ./common)

    # Common include directories
  set(RCCL_COMMON_INCLUDE_DIRS
    ${GTEST_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include # for generated rccl.h header
    ${PROJECT_BINARY_DIR}/hipify/src/include  # for rccl_bfloat16.h
    ${PROJECT_BINARY_DIR}/hipify/src/include/nccl_device  # for rccl_ptr.h
    ${PROJECT_BINARY_DIR}/hipify/gensrc # for rccl_bfloat16.h
    ${PROJECT_BINARY_DIR}/hipify/src # for graph/topo.h
    ${PROJECT_BINARY_DIR}/hipify/src/include/plugin # for recorder tests, nccl_tuner.h
    ${PROJECT_BINARY_DIR}/hipify/src/transport/net_ib_cast # for net_ib_fault_inject.h, net_ib_cast_inspect.h
    ${PROJECT_BINARY_DIR}/hipify/src/transport             # for net_ib_limits.h, net_ib/net_ib_flush_fault_inject.h
    ${ROCM_PATH}/include
    ${ROCM_PATH}
  )

  # Add MPI include directories if MPI tests are enabled
  if(ENABLE_MPI_TESTS AND MPI_CXX_INCLUDE_DIRS)
    list(APPEND RCCL_COMMON_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
  endif()

  # Common compile definitions
  set(RCCL_COMMON_COMPILE_DEFS ROCM_PATH="${ROCM_PATH}")
  if(LL128_ENABLED)
    list(APPEND RCCL_COMMON_COMPILE_DEFS ENABLE_LL128)
  endif()
  if(OPENMP_TESTS_ENABLED)
    list(APPEND RCCL_COMMON_COMPILE_DEFS ENABLE_OPENMP)
  endif()
  if(ENABLE_MPI_TESTS)
    list(APPEND RCCL_COMMON_COMPILE_DEFS MPI_TESTS_ENABLED)
    if(RCCL_HAS_RMA_IB_PROXY)
      list(APPEND RCCL_COMMON_COMPILE_DEFS RCCL_HAS_RMA_IB_PROXY)
    endif()
    if(ENABLE_HOST_API_TESTS)
      list(APPEND RCCL_COMMON_COMPILE_DEFS RCCL_ENABLE_HOST_API_TESTS)
    endif()
  endif()
  list(APPEND RCCL_COMMON_COMPILE_DEFS __HIP_PLATFORM_AMD__)

  # All test executables link against GTest; tell MPIHelpers.cpp so it can use
  # GTest's current_test_info() rather than relying on __has_include (which only
  # proves the header is installed, not that the target actually links the library).
  list(APPEND RCCL_COMMON_COMPILE_DEFS RCCL_MPIHelpers_HAS_GTEST)

  # Common link libraries
  set(RCCL_COMMON_LINK_LIBS
    ${GTEST_BOTH_LIBRARIES}
    hip::host hip::device hsa-runtime64::hsa-runtime64
    Threads::Threads
    dl
    fmt::fmt-header-only
  )

  # Link libs for the host-only micro-test binary (test/host). Deliberately no
  # HIP runtime on the link line, so any un-shimmed HIP API call surfaces as an
  # undefined-symbol link error instead of silently binding to libamdhip64.so.
  # HIP's *compile* requirements (device intrinsics, gfx codegen, headers) are
  # applied in test/host/CMakeLists.txt via $<TARGET_PROPERTY:hip::*,...> (a
  # CMake 3.16-safe stand-in for $<COMPILE_ONLY:hip::*>, which needs 3.27).
  set(MICRO_TEST_LINK_LIBS
    ${GTEST_BOTH_LIBRARIES}
    Threads::Threads
    dl
    fmt::fmt-header-only
  )
  if(OPENMP_TESTS_ENABLED)
    list(APPEND RCCL_COMMON_LINK_LIBS "${OpenMP_CXX_FLAGS}")
  endif()
  if(ENABLE_MPI_TESTS AND MPI_CXX_LIBRARIES)
    list(APPEND RCCL_COMMON_LINK_LIBS ${MPI_CXX_LIBRARIES})
  endif()

  # Get the compile definitions from the main rccl target
  # These helps to keep the test compile definitions in sync with the main rccl target
  # Also, all the structure layout remains the same across all the targets
  get_target_property(RCCL_COMPILE_DEFINITIONS rccl COMPILE_DEFINITIONS)
  if(RCCL_COMPILE_DEFINITIONS)
    list(APPEND RCCL_COMMON_COMPILE_DEFS ${RCCL_COMPILE_DEFINITIONS})
  endif()

  # Also get interface compile definitions
  get_target_property(RCCL_INTERFACE_COMPILE_DEFINITIONS rccl INTERFACE_COMPILE_DEFINITIONS)
  if(RCCL_INTERFACE_COMPILE_DEFINITIONS)
    list(APPEND RCCL_COMMON_COMPILE_DEFS ${RCCL_INTERFACE_COMPILE_DEFINITIONS})
  endif()

  # Collect testing framework source files
  set(TEST_SOURCE_FILES
    AllGatherTests.cpp
    AllReduceTests.cpp
    AllToAllTests.cpp
    AllToAllVTests.cpp
    AsymmetricVisibilityTests.cpp
    CollImplInfoTests.cpp
    BroadcastTests.cpp
    DoneEventOrderingTests.cpp
    GatherTests.cpp
    GroupCallTests.cpp
    NonBlockingTests.cpp
    RasJsonTests.cpp
    ReduceScatterTests.cpp
    ReduceTests.cpp
    RegisterTests.cpp
    ScatterTests.cpp
    P2pChannelScalingTests.cpp
    VersionGateTests.cpp
    SendRecvTests.cpp
    StandaloneTests.cpp
    DdaIpcTests.cpp
    NetPluginReloadTests.cpp
    plugin/net_reload_plugin.cpp
    ProfilerPluginCloseTests.cpp
    CpuAffinityTests.cpp
    TeardownStressTests.cpp
    _RecorderTests.cpp
    IbvSymbolsEnvTests.cpp
    common/main.cpp
    common/CallCollectiveForked.cpp
    common/CollectiveArgs.cpp
    common/EnvVars.cpp
    common/PrepDataFuncs.cpp
    common/PtrUnion.cpp
    common/ProcessIsolatedTestRunner.cpp
    common/TestBed.cpp
    common/TestBedChild.cpp
    common/StandaloneUtils.cpp
    proxy_trace/ProxyTraceUnitTests.cpp
    ../src/misc/proxy_trace/proxy_trace.cc
    latency_profiler/LatencyProfilerUnitTest.cpp
    ../src/misc/latency_profiler/CollTraceUtils.cc
    )

  # Due to default hidden symbol visibility, append source file if build type is not Debug.
  # It requires explicit addition of the following source file(s)
  # to the unit tests to ensure it is included for the existing rccl-UnitTests execution
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND TEST_SOURCE_FILES
      ../src/misc/recorder.cc
      ../src/misc/proxy_trace/proxy_trace.cc
      ../src/misc/ibvsymbols.cc  # buildIbvSymbols() is hidden in librccl; compile it in for IbvSymbolsEnvTests
    )
  endif()

  set(RCCL_TEST_EXECUTABLES rccl-UnitTests)

  # Create rccl-UnitTests binary
  add_executable(rccl-UnitTests ${TEST_SOURCE_FILES})

  # AICOMRCCL-1534 / net-plugin assignment tests: net_reload_plugin.cpp is compiled into
  # rccl-UnitTests and loaded in-process via NCCL_NET_PLUGIN=STATIC_PLUGIN (dlopen(NULL)).
  # Scope its headers to the file so the generic example headers don't shadow others.
  set_source_files_properties(plugin/net_reload_plugin.cpp PROPERTIES
    INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/../plugins/net/example/nccl)

  # dlsym(dlopen(NULL), ...) only finds symbols present in the dynamic symbol table;
  # export just the one plugin symbol (narrow, unlike a broad -rdynamic).
  target_link_options(rccl-UnitTests PRIVATE
    "LINKER:--export-dynamic-symbol=ncclNetPlugin_v12")

  # AICOMRCCL-2183: RcclMockFuncs.hpp's empty ncclDebugLog, needed to link the Release-only
  # sources above, overrides librccl.so's own in Debug and silences all library logging. Hidden
  # keeps it linkable here but out of .dynsym, as rccl-UnitTestsFixturesDebug does below.
  target_compile_options(rccl-UnitTests PRIVATE
    -fvisibility=hidden -fvisibility-inlines-hidden)

  # A real shared library, unlike the net plugin above: ProfilerPluginCloseTests checks
  # that RCCL unloads an external profiler plugin, so it needs a separately loadable
  # object. It lands beside rccl-UnitTests in the build tree and is installed beside it
  # too, so the test can find it by name relative to its own executable.
  add_library(rccl-test-profiler-stub SHARED plugin/profiler_stub_plugin.cpp)
  add_dependencies(rccl-UnitTests rccl-test-profiler-stub)
  target_compile_definitions(rccl-UnitTests PRIVATE
    RCCL_TEST_PROFILER_STUB_NAME="$<TARGET_FILE_NAME:rccl-test-profiler-stub>")

  # The test suite also runs from an install: TheRock CI executes the packaged
  # rccl-UnitTests with no build tree present, so the stub has to ship with it.
  # Deliberately not rocm_install(): that sends libraries to the library directory and
  # adds them to the package export set, while this stub has to ship beside the test
  # binaries that rocm_install() puts in CMAKE_INSTALL_BINDIR.
  install(TARGETS rccl-test-profiler-stub
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tests)

  # rccl-UnitTestsFixtures: Tests that only use header-only internal dependencies
  # (inline, static, constexpr, template functions) and can run in both Release and Debug.
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsFixtures)

    set(TEST_FIXTURE_SOURCE_FILES
      BitOpsTests.cpp
      IommuPassthrough_test.cpp
      MatchSubnet_test.cpp
      MiscTests.cpp
      RomeTopoConsensusTests.cpp
      EnqueueCountTests.cpp
      DdaCollCommonTests.cpp
      CheapPostSendFenceTests.cpp
      VersionInfoTests.cpp
      device/TestOp128.cpp
      device/TestTDM.cpp
      device/TestTdmCopy.cpp
      device/TestAsyncCopy.cpp
      device/TestLdsTransfer.cpp
      device/GinDeviceTests.cpp
      device/GinAnvilIpcDevice_test.cpp
      device/HierarchicalShuffleTests.cpp
      DeviceApiTests.cpp
      common/main_fixtures.cpp
      common/EnvVars.cpp
      common/ProcessIsolatedTestRunner.cpp
      common/TestChecks.cpp
      ../src/misc/kernel_config.cc
    )

  if(ENABLE_ROCSHMEM_GIN)
    list(APPEND TEST_FIXTURE_SOURCE_FILES
      gin/GinAnvilIpcTableHost_test.cpp
      ../src/gin/gin_anvil_ipc_table_host.cc
    )
  endif()

  if(ENABLE_ROCSHMEM_GIN AND (ENABLE_ROCSHMEM OR GIN_ANVIL_UNIT_TESTS))
    list(APPEND TEST_FIXTURE_SOURCE_FILES
      # GIN rocSHMEM-GDA device template test. Self-contained: supplies its own
      # rocshmem::QueuePair device stubs in-TU (no librocshmem device link, no
      # header shadow). Its body is guarded by NCCL_GIN_ROCSHMEM_GDA_ENABLE, which
      # gin_device_common.h sets when ENABLE_ROCSHMEM_GIN is defined on HIP.
      # Built here alongside the Anvil-SDMA template test under GIN_ANVIL_UNIT_TESTS
      # or when librccl is built with full ENABLE_ROCSHMEM.
      device/GinRocshmemGdaTemplate_test.cpp
      device/GinAnvilSdmaTemplate_test.cpp
    )
  endif()

    add_executable(rccl-UnitTestsFixtures ${TEST_FIXTURE_SOURCE_FILES})

    target_include_directories(rccl-UnitTestsFixtures PRIVATE
      ${PROJECT_BINARY_DIR}/hipify/src/device
      ${PROJECT_BINARY_DIR}/hipify/src/include
      ${PROJECT_BINARY_DIR}/hipify/src/include/nccl_device
    )

    # device/TestTdmCopy.cpp exercises tdm/tdmCopy.h, which detects TDM toolchain
    # support via __has_include(<hip/amd_detail/amd_gfx1250_TDM.h>). An SDK that omits
    # that descriptor header compiles every tdm:: entry point to a deleted stub and
    # IsTdmCopySupported() reports the GPU as unsupported, so warn rather than let the
    # TDM coverage silently degrade to no-ops.
    find_path(RCCL_GFX1250_TDM_HEADER
      NAMES hip/amd_detail/amd_gfx1250_TDM.h
      PATHS ${RCCL_COMMON_INCLUDE_DIRS} "${ROCM_PATH}/include")
    if(NOT RCCL_GFX1250_TDM_HEADER)
      message(WARNING "SDK lacks hip/amd_detail/amd_gfx1250_TDM.h; tdm:: entry points will compile to no-op stubs and device/TestTdmCopy.cpp will report TDM as unsupported")
    endif()

  if(ENABLE_ROCSHMEM_GIN)
    target_compile_definitions(rccl-UnitTestsFixtures PRIVATE ENABLE_ROCSHMEM_GIN)
  endif()

  # Match librccl.so ABI: ENABLE_ROCSHMEM changes ncclComm/ncclProxyState layout.
  # GIN CI builds librccl with ENABLE_ROCSHMEM_GIN only; device templates need
  # ENABLE_ROCSHMEM_GIN (above), not ENABLE_ROCSHMEM on this target.
  if(ENABLE_ROCSHMEM_GIN AND ENABLE_ROCSHMEM)
    target_compile_definitions(rccl-UnitTestsFixtures PRIVATE ENABLE_ROCSHMEM)
  endif()

  if(ENABLE_ROCSHMEM_GIN OR GIN_ANVIL_UNIT_TESTS)
    if(ROCSHMEM_INCLUDE_DIR)
      target_include_directories(rccl-UnitTestsFixtures PRIVATE ${ROCSHMEM_INCLUDE_DIR})
    elseif(ROCSHMEM_INSTALL_DIR)
      target_include_directories(rccl-UnitTestsFixtures PRIVATE
        "${ROCSHMEM_INSTALL_DIR}/include")
    endif()
    if(ROCSHMEM_SOURCE_DIR)
      target_include_directories(rccl-UnitTestsFixtures PRIVATE
        "${ROCSHMEM_SOURCE_DIR}/src"
        "${ROCSHMEM_SOURCE_DIR}/include")
    endif()
    if(ROCSHMEM_BUILD_DIR)
      target_include_directories(rccl-UnitTestsFixtures PRIVATE ${ROCSHMEM_BUILD_DIR})
    endif()
  endif()

  if(ENABLE_ROCSHMEM_GIN AND ROCM_VERSION VERSION_GREATER_EQUAL "60400")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsGinAnvilPlugin)

    add_executable(rccl-UnitTestsGinAnvilPlugin
      gin/GinAnvilPlugin_test.cpp
      gin/gin_anvil_plugin_test_stubs.cc
      ../src/gin/gin_plugin_anvil_sdma.cc
      ../src/gin/gin_anvil_ipc_table_host.cc
      common/main_fixtures.cpp
      common/EnvVars.cpp
      common/TestChecks.cpp
      common/ProcessIsolatedTestRunner.cpp
    )

    target_include_directories(rccl-UnitTestsGinAnvilPlugin PRIVATE
      ${PROJECT_BINARY_DIR}/hipify/src/include
      ${PROJECT_BINARY_DIR}/hipify/src/include/nccl_device
      ${CMAKE_CURRENT_SOURCE_DIR}/gin
    )
    if(ROCSHMEM_SOURCE_DIR)
      target_include_directories(rccl-UnitTestsGinAnvilPlugin PRIVATE
        ${ROCSHMEM_SOURCE_DIR}/include)
    else()
      target_include_directories(rccl-UnitTestsGinAnvilPlugin PRIVATE
        ${CMAKE_SOURCE_DIR}/../rocshmem/include)
    endif()

    target_compile_definitions(rccl-UnitTestsGinAnvilPlugin PRIVATE ENABLE_ROCSHMEM_GIN)
  endif()
  endif()

  # rccl-UnitTestsDevRuntime: self-contained host micro-test that #includes the
  # hipified src/dev_runtime.cc directly to reach its file-static symMemory*
  # helpers. Links no librccl.so; dependencies come from DevRuntimeTestsStubs.cc.
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsDevRuntime)

    add_executable(rccl-UnitTestsDevRuntime
      DevRuntimeTests.cpp
      DevRuntimeTestsStubs.cc
      host/fakes/nccl_fakes.cc         # shared reusable stubs for nccl* symbols
      common/main_fixtures.cpp
      common/EnvVars.cpp
      common/TestChecks.cpp
      common/ProcessIsolatedTestRunner.cpp
    )

    target_include_directories(rccl-UnitTestsDevRuntime PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/host/fakes
      ${PROJECT_BINARY_DIR}/hipify/src
      ${PROJECT_BINARY_DIR}/hipify/src/include
      ${PROJECT_BINARY_DIR}/hipify/src/include/nccl_device
      ${PROJECT_BINARY_DIR}/hipify/src/device
    )

    set_source_files_properties(DevRuntimeTests.cpp PROPERTIES
      COMPILE_DEFINITIONS
        "DEV_RUNTIME_CC_PATH=\"${PROJECT_BINARY_DIR}/hipify/src/dev_runtime.cc\"")

    # The included source and its headers live in the hipify-staged tree.
    add_dependencies(rccl-UnitTestsDevRuntime
      hipify_all copy_nccl_device_headers
      net_ib_header_symlinks rma_header_symlinks)
  endif()

  # GIN Anvil-SDMA put-segmentation host unit test (no GPU). Pure host C++: the
  # segment-math header (nccl_device/gin/anvil_sdma/gin_anvil_sdma_put_policy.h) is
  # compiled with GIN_SDMA_HOST_ONLY so it needs neither a built librccl nor device
  # codegen. It exercises the exact ginPutSegmentCount/ginPutSegmentAt the SDMA
  # backend Put drives on device. Runs under `ctest -L unit`.
  add_executable(rccl-UnitTestsGinSdmaPolicy gin/gin_sdma_policy_test.cpp)
  set_source_files_properties(gin/gin_sdma_policy_test.cpp PROPERTIES LANGUAGE CXX)
  target_include_directories(rccl-UnitTestsGinSdmaPolicy PRIVATE
    ${PROJECT_SOURCE_DIR}/src/include)
  target_compile_definitions(rccl-UnitTestsGinSdmaPolicy PRIVATE GIN_SDMA_HOST_ONLY)
  target_link_libraries(rccl-UnitTestsGinSdmaPolicy PRIVATE ${GTEST_BOTH_LIBRARIES} Threads::Threads)
  set_target_properties(rccl-UnitTestsGinSdmaPolicy PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test")
  add_test(NAME rccl-UnitTestsGinSdmaPolicy COMMAND rccl-UnitTestsGinSdmaPolicy)
  set_tests_properties(rccl-UnitTestsGinSdmaPolicy PROPERTIES LABELS "unit")

  # rccl-UnitTestsFixturesDebug: Tests that access internal symbols compiled into
  # librccl.so which are only visible in Debug builds (hidden visibility in Release).
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsFixturesDebug)

    set(TEST_FIXTURE_DEBUG_SOURCE_FILES
      AllocTests.cpp
      ParamTests.cpp
      ParameterApiTests.cpp
      ArgCheckTests.cpp
      BootstrapBidirTests.cpp
      ChannelDefaultsTests.cpp
      DdaAlltoAllThresholdTests.cpp
      DdaIpcEligibilityTests.cpp
      DdaFabricEligibilityTests.cpp
      CeAllReduceEligibilityTests.cpp
      CeAlltoAllvEligibilityTests.cpp
      EnqueueTests.cpp
      LL128RegVariantKernel_test.cpp
      IpcsocketTests.cpp
      TopoEnvPolicyTests.cpp
      P2pMaxNchannelsTests.cpp
      TopoNicFusionTests.cpp
      NetSocketTests.cpp
      PluginCompatV10_test.cpp
      PluginCompatV11_test.cpp
      ProxyTests.cpp
      PatAlgorithmTests.cpp
      RcclWrapTests.cpp
      SocketPollTests.cpp
      TransportTests.cpp
      TimeoutTests.cpp
      mem_manager/MemManagerTests.cpp
      graph/SearchTests.cpp
      graph/XmlTests.cpp
      graph/NetDevsPolicyP2pNetTests.cpp
      graph/TopoTests.cpp
      transport/NetIbCast/NetIbCastPlaneSubnetTests.cpp
      common/main_fixtures.cpp
      common/EnvVars.cpp
      common/ProcessIsolatedTestRunner.cpp
    )

    add_executable(rccl-UnitTestsFixturesDebug ${TEST_FIXTURE_DEBUG_SOURCE_FILES})

    # Hide the test executable's own symbols so it does NOT interpose
    # librccl.so's weak alloc.h template/global symbols (e.g.
    # ncclCudaHostCallocDebug<>, ncclDebugLevel) at load time. Without this,
    # those interposed-but-mismatched definitions corrupt the heap during a
    # full ncclCommInit* in this binary. The test still imports librccl's
    # internal (Debug-visible) symbols normally.
    target_compile_options(rccl-UnitTestsFixturesDebug PRIVATE
      -fvisibility=hidden -fvisibility-inlines-hidden)
    # NetDevsPolicyTests and NetDevsPolicyP2pNetTests load a topology XML fixture
    # SearchTests.cpp #includes the hipify-generated copy of src/graph/search.cc
    # directly (via #include SEARCH_CC_PATH) so it can exercise search.cc's
    # internal symbols in the same TU. Pass the absolute path as a source-scoped
    # define; the hipify step is ordered ahead of the test via the add_dependencies
    # / link to the rccl target that the common foreach below applies.
    set_source_files_properties(graph/SearchTests.cpp PROPERTIES
      COMPILE_DEFINITIONS
        "SEARCH_CC_PATH=\"${PROJECT_BINARY_DIR}/hipify/src/graph/search.cc\"")

    # TopoEnvPolicyTests and NetDevsPolicyP2pNetTests load a topology XML fixture
    # from the source tree (tools/topo_expl/models). Expose the test source dir
    # so they can build the path.
    target_compile_definitions(rccl-UnitTestsFixturesDebug PRIVATE
      RCCL_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

    # PluginCompatV10_test.cpp / PluginCompatV11_test.cpp resolve their in-process
    # mock plugin symbols via dlsym(dlopen(NULL), ...), which only sees the
    # executable's dynamic symbols.
    target_link_options(rccl-UnitTestsFixturesDebug PRIVATE
      -Wl,--export-dynamic-symbol=ncclNetPlugin_v10
      -Wl,--export-dynamic-symbol=ncclNetPlugin_v11
      -Wl,--export-dynamic-symbol=ncclCollNetPlugin_v11)
  endif()

  # rccl-UnitTestsAltRsmi: Compiles alt_rsmi.cc directly (not via librccl.so) with
  # ARSMI_TEST_BUILD, which gives external linkage to thread_local sysfs root variables
  # so tests can redirect them to a temp directory. No librccl.so symbols needed,
  # so this target works in both Release and Debug builds.
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsAltRsmi)

    set(TEST_ALTRSMI_SOURCE_FILES
      AltRsmiTests.cpp
      ../src/misc/alt_rsmi.cc
      common/main_altrsmi.cpp
      common/ProcessIsolatedTestRunner.cpp
    )

    add_executable(rccl-UnitTestsAltRsmi ${TEST_ALTRSMI_SOURCE_FILES})

    # Define ARSMI_TEST_BUILD specifically for rccl-UnitTestsAltRsmi
    target_compile_definitions(rccl-UnitTestsAltRsmi PRIVATE ARSMI_TEST_BUILD)

    # amdsmi_wrap.h is compiled directly into this target (via alt_rsmi.cc), so it
    # needs the amdsmi include path that the rccl library target gets but doesn't
    # propagate (PRIVATE scope). Without this, __has_include(<amd_smi/amdsmi.h>)
    # returns false in test TUs even when AMDSMI_FABRIC_DIRECT is set.
    if(DEFINED SMI_INCLUDE_DIR)
      target_include_directories(rccl-UnitTestsAltRsmi PRIVATE ${SMI_INCLUDE_DIR})
    endif()
  endif()

  # rccl-UnitTestsGdr: Compiles the peer-memory detection helper (gdr_peermem.cc,
  # shared by the net_ib and net_ib_cast transports) directly so the sysfs scan can
  # be exercised against a mock `memory_peers` tree in a temp directory. The helper
  # is pure (no HIP, no librccl.so symbols, no GPU), so this target builds in
  # Release and Debug.
  list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsGdr)

  set(TEST_GDR_SOURCE_FILES
    GdrPeerMemTests.cpp
    ../src/misc/gdr_peermem.cc
    common/main_altrsmi.cpp
  )

  add_executable(rccl-UnitTestsGdr ${TEST_GDR_SOURCE_FILES})
  # rccl-UnitTestsAmdSmi: tests for the amdsmi_wrap layer.
  #
  # AmdSmiFabricTests.cpp covers the fabric predicates and struct layout pinned in
  # amdsmi_wrap.h. Those are header-inline, so they need no librccl.so symbols and
  # run in both Release and Debug.
  #
  # AmdSmiWrapTests.cpp drives the amd_smi_* API itself, so it needs symbols that
  # src/CMakeLists.txt only exports under -fvisibility=default in Debug. Adding it
  # in Release would fail to link, so it joins the target only where they exist.
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsAmdSmi)

    # main_altrsmi.cpp is a plain InitGoogleTest/RUN_ALL_TESTS entry point, reused here.
    set(TEST_AMDSMI_SOURCE_FILES
      AmdSmiFabricTests.cpp
      common/main_altrsmi.cpp
    )

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
      list(APPEND TEST_AMDSMI_SOURCE_FILES
        AmdSmiWrapTests.cpp
        common/ProcessIsolatedTestRunner.cpp
      )

      # Loaded only by isolated lifecycle tests, never by the normal test process.
      add_library(rccl-AmdSmiLifecycleStub SHARED AmdSmiLifecycleStub.cpp)
      set_target_properties(rccl-AmdSmiLifecycleStub PROPERTIES
        OUTPUT_NAME "amd_smi"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/amdsmi_lifecycle_stub"
      )

      # Match the versioned name used by amdsmi_wrap when headers are present.
      # Older ROCm configurations may not define SMI_INCLUDE_DIR, so also check
      # ROCM_PATH; without either header amdsmi_wrap uses the unversioned name.
      if(EXISTS "${SMI_INCLUDE_DIR}/amd_smi/amdsmi.h")
        set(AMDSMI_TEST_HEADER "${SMI_INCLUDE_DIR}/amd_smi/amdsmi.h")
      elseif(EXISTS "${ROCM_PATH}/include/amd_smi/amdsmi.h")
        set(AMDSMI_TEST_HEADER "${ROCM_PATH}/include/amd_smi/amdsmi.h")
      endif()
      if(AMDSMI_TEST_HEADER)
        file(STRINGS "${AMDSMI_TEST_HEADER}" AMDSMI_VERSION_MAJOR_LINE
          REGEX "^#define AMDSMI_LIB_VERSION_MAJOR [0-9]+"
          LIMIT_COUNT 1
        )
        string(REGEX REPLACE ".* ([0-9]+)$" "\\1"
          AMDSMI_TEST_SOVERSION "${AMDSMI_VERSION_MAJOR_LINE}"
        )
        set_target_properties(rccl-AmdSmiLifecycleStub PROPERTIES
          SOVERSION "${AMDSMI_TEST_SOVERSION}"
        )
      endif()
    endif()

    add_executable(rccl-UnitTestsAmdSmi ${TEST_AMDSMI_SOURCE_FILES})

    if(CMAKE_BUILD_TYPE MATCHES "Debug")
      add_dependencies(rccl-UnitTestsAmdSmi rccl-AmdSmiLifecycleStub)
      target_compile_definitions(rccl-UnitTestsAmdSmi PRIVATE
        AMDSMI_TEST_STUB_DIR="$<TARGET_FILE_DIR:rccl-AmdSmiLifecycleStub>"
        AMDSMI_TEST_STUB_SONAME="$<TARGET_SONAME_FILE_NAME:rccl-AmdSmiLifecycleStub>"
      )
    endif()

    # amdsmi_wrap.h needs both halves of this pairing: the include path so its
    # __has_include finds amdsmi, and AMDSMI_FABRIC_DIRECT, which is PRIVATE to the
    # rccl target and so does not reach test TUs. With the headers reachable but the
    # define absent, the header's fabric compat block compiles on top of the real
    # declarations and every fabric type collides.
    if(DEFINED SMI_INCLUDE_DIR)
      target_include_directories(rccl-UnitTestsAmdSmi PRIVATE ${SMI_INCLUDE_DIR})
    endif()
    if(AMDSMI_FABRIC_API)
      target_compile_definitions(rccl-UnitTestsAmdSmi PRIVATE AMDSMI_FABRIC_DIRECT)
    endif()
  endif()

  # Build the GIN example plugin for host-API custom-plugin regression tests.
  if(ENABLE_MPI_TESTS AND ENABLE_HOST_API_TESTS)
    set(RCCL_GIN_EXAMPLE_PLUGIN_DIR "${CMAKE_BINARY_DIR}/test/unit/plugins")
    set(RCCL_GIN_EXAMPLE_PLUGIN "${RCCL_GIN_EXAMPLE_PLUGIN_DIR}/libnccl-gin-example.so")
    find_program(RCCL_GIN_EXAMPLE_CC NAMES gcc cc REQUIRED)
    add_custom_command(
      OUTPUT "${RCCL_GIN_EXAMPLE_PLUGIN}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${RCCL_GIN_EXAMPLE_PLUGIN_DIR}"
      COMMAND ${RCCL_GIN_EXAMPLE_CC}
              -I${PROJECT_SOURCE_DIR}/plugins/gin/example/nccl
              -fPIC -shared
              -o "${RCCL_GIN_EXAMPLE_PLUGIN}"
              ${PROJECT_SOURCE_DIR}/plugins/gin/example/plugin.c
      DEPENDS ${PROJECT_SOURCE_DIR}/plugins/gin/example/plugin.c
      COMMENT "Building libnccl-gin-example.so for Host API regression tests"
      VERBATIM
    )
    add_custom_target(nccl-gin-example DEPENDS "${RCCL_GIN_EXAMPLE_PLUGIN}")
  endif()

  # Create separate MPI test binary if MPI tests are enabled
  if(ENABLE_MPI_TESTS)
    set(MPI_TEST_SOURCE_FILES
      common/main_mpi.cpp
      common/MPIHelpers.cpp
      common/MPITestCore.cpp
      common/MPIEnvironment.cpp
      common/TestChecks.cpp
      transport/TransportMPIBase.cpp
      transport/P2pMPITests.cpp
      transport/NetMPITests.cpp
      transport/ShmMPITests.cpp
      transport/NetIbMPI/GeneralTests.cpp
      transport/NetIbMPI/GdrFlushTests.cpp
      transport/NetIbMPI/NicFusionTests.cpp
      transport/NetIbMPI/CastTests.cpp
      transport/NetIbMPI/FaultInjectTests.cpp
      transport/NetIbMPI/OpsFaultUnitTests.cpp
      transport/RmaMPI/RmaMPITests.cpp
      transport/NetIbMPI/StressTests.cpp
      transport/GinDeviceMPITests.cpp
      WarpSpeedMPITests.cpp
      ImplicitLaunchOrderMPITests.cpp
      CommMPITests.cpp
      NChannelsPerNetPeerMPITests.cpp
      RegistrationMPITests.cpp
      HostApiMPITests.cpp
      HostApiCustomGinPluginMPITests.cpp
      RmaExternalPluginPutSignalMPITests.cpp
      plugin/net_reload_plugin.cpp
      mem_manager/SuspendResumeMPITests.cpp
      RevokeMPITests.cpp
      ce/CeMPITests.cpp
      ce/CeInternalMPITests.cpp
      GrowMPITests.cpp
      GinNcclTimeoutMPITests.cpp
      InspectorPromP2pMPITests.cpp
      SymmetricKernelMPITests.cpp
      SymmetricWindowMPITests.cpp
      DeviceApiMPITests.cpp
      CommNetMismatchMPITests.cpp
      test_sparse_send_recv.cpp
      DdaAllReduceMPITests.cpp
      SymmetricAbortCheckModeMPITests.cpp
    )

    add_executable(rccl-UnitTestsMPI ${MPI_TEST_SOURCE_FILES})

    # plugin/net_reload_plugin.cpp is compiled into rccl-UnitTestsMPI so
    # RmaExternalPluginPutSignalMPITests can load its ncclRmaPlugin_v13 in-process
    # via STATIC_PLUGIN (dlopen(NULL))
    target_link_options(rccl-UnitTestsMPI PRIVATE
      "LINKER:--export-dynamic-symbol=ncclGinPlugin_v13"
      "LINKER:--export-dynamic-symbol=ncclRmaPlugin_v13")

    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsMPI)

    # Hand the in-tree Inspector plugin to InspectorPromP2pMPITests so it can
    # load a profiler without the caller setting NCCL_INSPECTOR_PLUGIN_SO.
    if(TARGET rccl-profiler-inspector)
      add_dependencies(rccl-UnitTestsMPI rccl-profiler-inspector)
      target_compile_definitions(rccl-UnitTestsMPI PRIVATE
        RCCL_INSPECTOR_PLUGIN_SO="$<TARGET_FILE:rccl-profiler-inspector>")
    endif()

    if(ENABLE_HOST_API_TESTS)
      add_dependencies(rccl-UnitTestsMPI nccl-gin-example)
      target_compile_definitions(rccl-UnitTestsMPI PRIVATE
        RCCL_GIN_EXAMPLE_PLUGIN_DIR="${RCCL_GIN_EXAMPLE_PLUGIN_DIR}"
      )

      # Minimal MPI binary for NCCL 2.30.7 custom-GIN-plugin regression only.
      # Avoids compiling the full rccl-UnitTestsMPI suite, which may lag API
      # updates on in-progress NCCL sync branches.
      add_executable(rccl-HostApiCustomGinPluginMPI
        common/main_mpi.cpp
        common/MPIHelpers.cpp
        common/MPITestCore.cpp
        common/MPIEnvironment.cpp
        common/TestChecks.cpp
        common/MPIHostUtils.cc
        HostApiCustomGinPluginMPITests.cpp
      )
      list(APPEND RCCL_TEST_EXECUTABLES rccl-HostApiCustomGinPluginMPI)
      add_dependencies(rccl-HostApiCustomGinPluginMPI nccl-gin-example)
      target_compile_definitions(rccl-HostApiCustomGinPluginMPI PRIVATE
        RCCL_GIN_EXAMPLE_PLUGIN_DIR="${RCCL_GIN_EXAMPLE_PLUGIN_DIR}"
      )
    endif()
  endif()

  foreach(test_executable IN LISTS RCCL_TEST_EXECUTABLES)
    target_include_directories(${test_executable} PRIVATE ${RCCL_COMMON_INCLUDE_DIRS})
    target_compile_definitions(${test_executable} PRIVATE ${RCCL_COMMON_COMPILE_DEFS})
    if(ENABLE_DEVICE_LINKER)
      # Mirror the rccl library define so device-linker-specific test expectations compile in.
      target_compile_definitions(${test_executable} PRIVATE RCCL_DEVICE_LINKER)
    endif()
    target_link_libraries(${test_executable} PRIVATE ${RCCL_COMMON_LINK_LIBS})
    if(BUILD_ADDRESS_SANITIZER)
      target_compile_options(${test_executable} PRIVATE -fsanitize=address)
      target_link_options(${test_executable} PRIVATE -fsanitize=address -shared-libasan)
      if(DEFINED ASAN_RUNTIME_DIR)
        target_link_options(${test_executable} PRIVATE "LINKER:-rpath,${ASAN_RUNTIME_DIR}")
      endif()
    endif()

    # Add MPI-specific configuration if MPI tests are enabled
    if(ENABLE_MPI_TESTS)
      if(MPI_CXX_COMPILE_FLAGS)
        target_compile_options(${test_executable} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
      endif()
      if(MPI_CXX_LINK_FLAGS)
        set_target_properties(${test_executable} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
      endif()
    endif()
    # rccl-UnitTestsGinAnvilPlugin is a self-contained host unit test: it compiles
    # the plugin sources directly and provides its own stubs (bootstrap/devr/debug/
    # factory), so it must NOT link librccl.so. When RCCL is built with
    # ENABLE_ROCSHMEM_GIN=ON but ENABLE_ROCSHMEM=OFF, librccl.so carries unresolved
    # rocSHMEM symbols (e.g. rocshmem::envvar::log_flags) that would fail at load
    # time for any executable linking it.
    if(test_executable STREQUAL "rccl-UnitTestsGinAnvilPlugin" OR
       test_executable STREQUAL "rccl-UnitTestsDevRuntime")
      # Intentionally no rccl link; needed symbols come from the compiled-in
      # source(s) + stubs and RCCL_COMMON_LINK_LIBS (hip/hsa/gtest/dl).
    elseif(BUILD_SHARED_LIBS)
      target_link_libraries(${test_executable} PRIVATE rccl)
      if(${HOST_OS_ID} STREQUAL "debian")
        set_property(TARGET ${test_executable} PROPERTY INSTALL_RPATH "${CMAKE_BINARY_DIR}")
      elseif(DEFINED HOST_OS_FAMILY AND "${HOST_OS_FAMILY}" STREQUAL "debian")
        set_property(TARGET ${test_executable} PROPERTY INSTALL_RPATH "${CMAKE_BINARY_DIR}")
      endif()
    else()
      add_dependencies(${test_executable} rccl)
      target_link_libraries(${test_executable} PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR} -lrocm_smi64 -L${ROCM_PATH}/lib -L${ROCM_PATH}/rocm_smi/lib)
    endif()

    # Link the GIN device backend library and device-link its rdc code (-fgpu-rdc
    # --hip-link --offload-arch=... -rdynamic) so GIN device tests can run.
    if(ENABLE_ROCSHMEM_GIN AND ROCSHMEM_INSTALL_DIR AND "${test_executable}" STREQUAL "rccl-UnitTestsMPI")
      set(_ROCSHMEM_OFFLOAD_FLAGS "")
      foreach(_g IN LISTS GPU_TARGETS)
        if(_g)
          list(APPEND _ROCSHMEM_OFFLOAD_FLAGS "--offload-arch=${_g}")
        endif()
      endforeach()
      # Fall back to linking by bare name if the hint lookup misses (avoids injecting <VAR>-NOTFOUND into the link line).
      find_library(_ROCSHMEM_DRM        NAMES drm        HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib)
      find_library(_ROCSHMEM_DRM_AMDGPU NAMES drm_amdgpu HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib)
      if(NOT _ROCSHMEM_DRM)
        set(_ROCSHMEM_DRM drm)
      endif()
      if(NOT _ROCSHMEM_DRM_AMDGPU)
        set(_ROCSHMEM_DRM_AMDGPU drm_amdgpu)
      endif()
      # Enable the GIN device backend in the test kernels so ncclGinCall has a
      # dispatch case for the corresponding runtime GIN type.
      target_compile_definitions(${test_executable} PRIVATE NCCL_GIN_ROCSHMEM_GDA_ENABLE=1)
      target_compile_options(${test_executable} PRIVATE -fgpu-rdc)
      target_link_libraries(${test_executable} PRIVATE
          ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a
          ibverbs hsa-runtime64 hsakmt numa ${_ROCSHMEM_DRM} ${_ROCSHMEM_DRM_AMDGPU})
      target_link_options(${test_executable} PRIVATE -fgpu-rdc --hip-link ${_ROCSHMEM_OFFLOAD_FLAGS} -rdynamic)
    endif()

    # librccl.so built with ENABLE_ROCSHMEM_GIN but without host librocshmem
    # (ENABLE_ROCSHMEM=OFF, e.g. gin CI --rocshmem-gin) leaves
    # rocshmem::envvar::log_flags undefined at load time. Link host rocSHMEM
    # into the Fixtures binary so the executable satisfies those symbols when
    # the dynamic loader brings in librccl.so.
    if(ENABLE_ROCSHMEM_GIN AND GIN_ANVIL_UNIT_TESTS AND NOT ENABLE_ROCSHMEM
       AND ROCSHMEM_INSTALL_DIR AND "${test_executable}" STREQUAL "rccl-UnitTestsFixtures")
      set(_ROCSHMEM_FIXTURES_OFFLOAD_FLAGS "")
      foreach(_g IN LISTS GPU_TARGETS)
        if(_g)
          list(APPEND _ROCSHMEM_FIXTURES_OFFLOAD_FLAGS "--offload-arch=${_g}")
        endif()
      endforeach()
      find_library(_ROCSHMEM_FIXTURES_DRM        NAMES drm        HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib)
      find_library(_ROCSHMEM_FIXTURES_DRM_AMDGPU NAMES drm_amdgpu HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib)
      if(NOT _ROCSHMEM_FIXTURES_DRM)
        set(_ROCSHMEM_FIXTURES_DRM drm)
      endif()
      if(NOT _ROCSHMEM_FIXTURES_DRM_AMDGPU)
        set(_ROCSHMEM_FIXTURES_DRM_AMDGPU drm_amdgpu)
      endif()
      target_link_libraries(${test_executable} PRIVATE
        ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a
        ibverbs hsa-runtime64 hsakmt numa
        ${_ROCSHMEM_FIXTURES_DRM} ${_ROCSHMEM_FIXTURES_DRM_AMDGPU})
      target_link_options(${test_executable} PRIVATE
        -fgpu-rdc --hip-link ${_ROCSHMEM_FIXTURES_OFFLOAD_FLAGS} -rdynamic)
    endif()

    rocm_install(TARGETS ${test_executable} COMPONENT tests)
  endforeach()

  # Suite H: shadow rocSHMEM sdma headers with no-op test stubs (higher-priority include path).
  if(ENABLE_ROCSHMEM_GIN AND (ENABLE_ROCSHMEM OR GIN_ANVIL_UNIT_TESTS))
    if(ROCSHMEM_SOURCE_DIR)
      target_include_directories(rccl-UnitTestsFixtures PRIVATE ${ROCSHMEM_SOURCE_DIR}/src)
    else()
      target_include_directories(rccl-UnitTestsFixtures PRIVATE ${CMAKE_SOURCE_DIR}/../rocshmem/src)
    endif()
    target_include_directories(rccl-UnitTestsFixtures BEFORE PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/device)
  endif()

  # Create install-time test file for distribution
  set(INSTALL_TEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
  file(WRITE "${INSTALL_TEST_FILE}"
  [=[
  # This is a test file generated by rccl for install time.
  # Tests are defined with relative paths to work in the installed location.
  ]=]
  )

  # Shared ROCm Libraries CTest categories (see shared/ctest/README.md)
  if(ROCM_SYSTEMS_ROOT AND EXISTS "${ROCM_SYSTEMS_ROOT}/shared/ctest/TestCategories.cmake")
    # Currently these files only exist in rocm-libraries, without that repo this will fail.
    # get_filename_component(ROCM_SYSTEMS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    include("${ROCM_SYSTEMS_ROOT}/shared/ctest/TestCategories.cmake")
    enable_testing()

    set(_rccl_test_categories_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
    if(EXISTS "${_rccl_test_categories_yaml}")
        message(STATUS "Applying test categories for rccl-UnitTests")
        apply_test_category_labels(
            rccl-UnitTests
            "${_rccl_test_categories_yaml}"
            "${PROJECT_BINARY_DIR}"
            "${INSTALL_TEST_FILE}"
        )
    else()
        message(WARNING "Skipping test categories for rccl-UnitTests: missing ${_rccl_test_categories_yaml}")
    endif()

    if(ROCM_VERSION VERSION_GREATER_EQUAL "60400")
        set(_rccl_test_categories_fixtures_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_fixtures.yaml")
        if(EXISTS "${_rccl_test_categories_fixtures_yaml}")
            message(STATUS "Applying test categories for rccl-UnitTestsFixtures")
            apply_test_category_labels(
                rccl-UnitTestsFixtures
                "${_rccl_test_categories_fixtures_yaml}"
                "${PROJECT_BINARY_DIR}"
                "${INSTALL_TEST_FILE}"
            )
        else()
            message(WARNING "Skipping test categories for rccl-UnitTestsFixtures: missing ${_rccl_test_categories_fixtures_yaml}")
        endif()

        # Registered outside the Debug-only block below: this target's fabric predicate
        # and layout cases need no librccl symbols and so run in Release too. The
        # AmdSmiWrap* patterns in the yaml match nothing in a Release binary, where
        # that file is left out for lack of exported amd_smi_* symbols.
        set(_rccl_test_categories_amdsmi_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_amdsmi.yaml")
        if(EXISTS "${_rccl_test_categories_amdsmi_yaml}")
            message(STATUS "Applying test categories for rccl-UnitTestsAmdSmi")
            apply_test_category_labels(
                rccl-UnitTestsAmdSmi
                "${_rccl_test_categories_amdsmi_yaml}"
                "${PROJECT_BINARY_DIR}"
                "${INSTALL_TEST_FILE}"
            )
        else()
            message(WARNING "Skipping test categories for rccl-UnitTestsAmdSmi: missing ${_rccl_test_categories_amdsmi_yaml}")
        endif()
    endif()

    if(ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
        set(_rccl_test_categories_fixtures_debug_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_fixtures_debug.yaml")
        if(EXISTS "${_rccl_test_categories_fixtures_debug_yaml}")
            message(STATUS "Applying test categories for rccl-UnitTestsFixturesDebug")
            apply_test_category_labels(
                rccl-UnitTestsFixturesDebug
                "${_rccl_test_categories_fixtures_debug_yaml}"
                "${PROJECT_BINARY_DIR}"
                "${INSTALL_TEST_FILE}"
            )
        else()
            message(WARNING "Skipping test categories for rccl-UnitTestsFixturesDebug: missing ${_rccl_test_categories_fixtures_debug_yaml}")
        endif()

        set(_rccl_test_categories_altrsmi_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_altrsmi.yaml")
        if(EXISTS "${_rccl_test_categories_altrsmi_yaml}")
            message(STATUS "Applying test categories for rccl-UnitTestsAltRsmi")
            apply_test_category_labels(
                rccl-UnitTestsAltRsmi
                "${_rccl_test_categories_altrsmi_yaml}"
                "${PROJECT_BINARY_DIR}"
                "${INSTALL_TEST_FILE}"
            )
        else()
            message(WARNING "Skipping test categories for rccl-UnitTestsAltRsmi: missing ${_rccl_test_categories_altrsmi_yaml}")
        endif()
    endif()

    if(ENABLE_MPI_TESTS)
        set(_rccl_test_categories_mpi_yaml "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_mpi.yaml")
        if(EXISTS "${_rccl_test_categories_mpi_yaml}")
            message(STATUS "Applying test categories for rccl-UnitTestsMPI")
            apply_test_category_labels(
                rccl-UnitTestsMPI
                "${_rccl_test_categories_mpi_yaml}"
                "${PROJECT_BINARY_DIR}"
            )
        else()
            message(WARNING "Skipping test categories for rccl-UnitTestsMPI: missing ${_rccl_test_categories_mpi_yaml}")
        endif()
    endif()
  endif()

  install(
    FILES "${INSTALL_TEST_FILE}"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/${PROJECT_NAME}"
    COMPONENT tests
    RENAME "CTestTestfile.cmake"
  )

  # Host-only micro-test binaries (rccl-UnitTestsMicro). The host/ CMakeLists
  # is dual-mode: under BUILD_TESTS it builds only the in-RCCL-build micro-test
  # target and returns; invoked standalone it builds rccl-HostUnitTests too.
  add_subdirectory(host)

endif()
