# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

cmake_minimum_required(VERSION 3.16)

if(BUILD_TESTS)

  option(OPENMP_TESTS_ENABLED "Enable OpenMP for unit tests" OFF)
  option(ENABLE_MPI_TESTS "Enable MPI-based tests" OFF)
  option(RCCL_HAS_GIN_IB_PROXY "Build the IB Proxy GIN transport-level tests" OFF)
  option(ENABLE_HOST_API_TESTS "Build the host-API tests" OFF)


  message("Building rccl unit tests (Installed in /test/rccl-UnitTests)")
  if(ENABLE_MPI_TESTS)
    message("MPI-based tests are enabled")
  endif()

  if (ENABLE_CODE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
    set(HIPCC_COMPILE_FLAGS "${HIPCC_COMPILE_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
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
    ${PROJECT_BINARY_DIR}/hipify/src/transport             # for net_ib_limits.h
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
    if(RCCL_HAS_GIN_IB_PROXY)
      list(APPEND RCCL_COMMON_COMPILE_DEFS RCCL_HAS_GIN_IB_PROXY)
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
    BroadcastTests.cpp
    GatherTests.cpp
    GroupCallTests.cpp
    NonBlockingTests.cpp
    ReduceScatterTests.cpp
    ReduceTests.cpp
    RegisterTests.cpp
    ScatterTests.cpp
    SendRecvTests.cpp
    StandaloneTests.cpp
    _RecorderTests.cpp
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
    )
  endif()

  set(RCCL_TEST_EXECUTABLES rccl-UnitTests)

  # Create rccl-UnitTests binary
  add_executable(rccl-UnitTests ${TEST_SOURCE_FILES})

  # rccl-UnitTestsFixtures: Tests that only use header-only internal dependencies
  # (inline, static, constexpr, template functions) and can run in both Release and Debug.
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsFixtures)

    set(TEST_FIXTURE_SOURCE_FILES
      BitOpsTests.cpp
      MiscTests.cpp
      RomeTopoConsensusTests.cpp
      EnqueueCountTests.cpp
      device/TestOp128.cpp
      device/GinDeviceTests.cpp
      common/main_fixtures.cpp
      common/EnvVars.cpp
      common/ProcessIsolatedTestRunner.cpp
      common/TestChecks.cpp
    )

    add_executable(rccl-UnitTestsFixtures ${TEST_FIXTURE_SOURCE_FILES})

    target_include_directories(rccl-UnitTestsFixtures PRIVATE
      ${PROJECT_BINARY_DIR}/hipify/src/device
    )
  endif()

  # rccl-UnitTestsFixturesDebug: Tests that access internal symbols compiled into
  # librccl.so which are only visible in Debug builds (hidden visibility in Release).
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsFixturesDebug)

    set(TEST_FIXTURE_DEBUG_SOURCE_FILES
      AllocTests.cpp
      ParamTests.cpp
      ArgCheckTests.cpp
      BootstrapBidirTests.cpp
      EnqueueTests.cpp
      IpcsocketTests.cpp
      NetSocketTests.cpp
      ProxyTests.cpp
      RcclWrapTests.cpp
      TransportTests.cpp
      mem_manager/MemManagerTests.cpp
      graph/XmlTests.cpp
      common/main_fixtures.cpp
      common/EnvVars.cpp
      common/ProcessIsolatedTestRunner.cpp
    )

    add_executable(rccl-UnitTestsFixturesDebug ${TEST_FIXTURE_DEBUG_SOURCE_FILES})
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
      transport/NetIbMPI/NicFusionTests.cpp
      transport/NetIbMPI/CastTests.cpp
      transport/NetIbMPI/FaultInjectTests.cpp
      transport/GinMPI/GinMPITests.cpp
      transport/NetIbMPI/StressTests.cpp
      transport/GinDeviceMPITests.cpp
      ImplicitLaunchOrderMPITests.cpp
      CommMPITests.cpp
      RegistrationMPITests.cpp
      HostApiMPITests.cpp
      mem_manager/SuspendResumeMPITests.cpp
      RevokeMPITests.cpp
      ce/CeMPITests.cpp
      ce/CeInternalMPITests.cpp
      GrowMPITests.cpp
    )

    add_executable(rccl-UnitTestsMPI ${MPI_TEST_SOURCE_FILES})
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsMPI)
  endif()

  foreach(test_executable IN LISTS RCCL_TEST_EXECUTABLES)
    target_include_directories(${test_executable} PRIVATE ${RCCL_COMMON_INCLUDE_DIRS})
    target_compile_definitions(${test_executable} PRIVATE ${RCCL_COMMON_COMPILE_DEFS})
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
    if(BUILD_SHARED_LIBS)
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

    rocm_install(TARGETS ${test_executable} COMPONENT tests)
  endforeach()

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

endif()
