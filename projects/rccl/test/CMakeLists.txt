# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

cmake_minimum_required(VERSION 3.16)

if(BUILD_TESTS)

  option(OPENMP_TESTS_ENABLED "Enable OpenMP for unit tests" OFF)

  message("Building rccl unit tests (Installed in /test/rccl-UnitTests)")

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

  include_directories(${GTEST_INCLUDE_DIRS} ./common)

    # Common include directories
  set(RCCL_COMMON_INCLUDE_DIRS
    ${GTEST_INCLUDE_DIRS}
    ${PROJECT_BINARY_DIR}/include # for generated rccl.h header
    ${PROJECT_BINARY_DIR}/hipify/src/include  # for rccl_bfloat16.h
    ${PROJECT_BINARY_DIR}/hipify/gensrc # for rccl_bfloat16.h
    ${PROJECT_BINARY_DIR}/hipify/src # for graph/topo.h
    ${PROJECT_BINARY_DIR}/hipify/src/include/plugin # for recorder tests, nccl_tuner.h
    ${ROCM_PATH}/include
    ${ROCM_PATH}
  )

  # Common compile definitions
  set(RCCL_COMMON_COMPILE_DEFS ROCM_PATH="${ROCM_PATH}")
  if(LL128_ENABLED)
    list(APPEND RCCL_COMMON_COMPILE_DEFS ENABLE_LL128)
  endif()
  if(OPENMP_TESTS_ENABLED)
    list(APPEND RCCL_COMMON_COMPILE_DEFS ENABLE_OPENMP)
  endif()
  list(APPEND RCCL_COMMON_COMPILE_DEFS __HIP_PLATFORM_AMD__)

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
  
  # Create rccl-UnitTestsFixtures binary if ROCm version is 4.6.0 or greater
  # and build type is Debug
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")
    add_dependencies(rccl-UnitTests replace_static_in_hipify)
    
    # Add rccl-UnitTestsFixtures binary
    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsFixtures)

    set(TEST_FIXTURE_SOURCE_FILES
      AltRsmiTests.cpp
      AllocTests.cpp
      ParamTests.cpp
      ArgCheckTests.cpp
      BitOpsTests.cpp
      CollRegTests.cpp
      CommTests.cpp
      EnqueueTests.cpp
      IpcsocketTests.cpp
      NetSocketTests.cpp
      P2pTests.cpp
      ProxyTests.cpp
      RcclWrapTests.cpp
      ShmTests.cpp
      TransportTests.cpp
      common/main_fixtures.cpp
      common/EnvVars.cpp
      graph/XmlTests.cpp
    )

    add_executable(rccl-UnitTestsFixtures ${TEST_FIXTURE_SOURCE_FILES})
    add_dependencies(rccl-UnitTestsFixtures replace_static_in_hipify)
  endif()

  foreach(test_executable IN LISTS RCCL_TEST_EXECUTABLES)
    target_include_directories(${test_executable} PRIVATE ${RCCL_COMMON_INCLUDE_DIRS})
    target_compile_definitions(${test_executable} PRIVATE ${RCCL_COMMON_COMPILE_DEFS})
    target_link_libraries(${test_executable} PRIVATE ${RCCL_COMMON_LINK_LIBS})
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

endif()

