# Integration tests (require GPU or GPU-specific encoding)

#===----------------------------------------------------------------------===//
# GoogleTest-based integration tests
#===----------------------------------------------------------------------===//

#===----------------------------------------------------------------------===//
# GPU hardware integration tests (require actual AMD GPU)
#===----------------------------------------------------------------------===//

if(AEGISBIT_HAS_GPU)
  # GPU launch and trace buffer tests
  add_executable(gpu_launch_gtest GPULaunchGTest.cpp)
  target_link_libraries(gpu_launch_gtest
    aegisbit
    GTest::gtest_main
  )
  target_include_directories(gpu_launch_gtest PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    ${LLVM_INCLUDE_DIRS}
  )
  add_test(NAME GPULaunchGTest COMMAND gpu_launch_gtest)

  set_tests_properties(GPULaunchGTest PROPERTIES
    ENVIRONMENT "GTEST_COLOR=1"
    LABELS "gpu"
  )

  #---------------------------------------------------------------------------
  # Find hipcc and detect GPU architecture (must happen before any hipcc usage)
  #---------------------------------------------------------------------------
  find_program(HIPCC_EXECUTABLE hipcc
    PATHS
      "${ROCM_PATH}/bin"
      "/opt/rocm/bin"
    NO_DEFAULT_PATH
  )

  if(HIPCC_EXECUTABLE)
    # Auto-detect GPU architecture from rocminfo
    find_program(ROCMINFO rocminfo PATHS "${ROCM_PATH}/bin" "/opt/rocm/bin" NO_DEFAULT_PATH)
    if(ROCMINFO)
      execute_process(
        COMMAND ${ROCMINFO}
        OUTPUT_VARIABLE ROCMINFO_OUTPUT
        ERROR_QUIET
      )
      string(REGEX MATCH "gfx[0-9a-z]+" DETECTED_GPU_ARCH "${ROCMINFO_OUTPUT}")
      if(DETECTED_GPU_ARCH)
        set(TEST_GPU_ARCH "${DETECTED_GPU_ARCH}" CACHE STRING "GPU architecture for tests" FORCE)
        message(STATUS "Auto-detected GPU architecture: ${TEST_GPU_ARCH}")
      else()
        set(TEST_GPU_ARCH "gfx942" CACHE STRING "GPU architecture for tests")
        message(STATUS "Could not detect GPU, defaulting to: ${TEST_GPU_ARCH}")
      endif()
    else()
      set(TEST_GPU_ARCH "gfx942" CACHE STRING "GPU architecture for tests")
      message(STATUS "rocminfo not found, defaulting to: ${TEST_GPU_ARCH}")
    endif()

    # Common include flags for all hipcc-compiled tests
    set(GTEST_INC_DIR "${CMAKE_BINARY_DIR}/_deps/googletest-src/googletest/include")

    set(HIPCC_INCLUDE_FLAGS
      -I${CMAKE_SOURCE_DIR}/include
      -I${CMAKE_CURRENT_SOURCE_DIR}/fixtures
      -I${GTEST_INC_DIR}
    )

    foreach(LLVM_INC_DIR ${LLVM_INCLUDE_DIRS})
      list(APPEND HIPCC_INCLUDE_FLAGS -I${LLVM_INC_DIR})
    endforeach()

    #---------------------------------------------------------------------------
    # HSA Pool GPU Tests - Critical gate tests for trace buffer approach
    #---------------------------------------------------------------------------
    set(HSA_POOL_GPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/HSAPoolGPUGTest.cpp")
    set(HSA_POOL_GPU_OBJ "${CMAKE_CURRENT_BINARY_DIR}/HSAPoolGPUGTest.o")

    add_custom_command(
      OUTPUT ${HSA_POOL_GPU_OBJ}
      COMMAND ${HIPCC_EXECUTABLE}
        -c
        --offload-arch=${TEST_GPU_ARCH}
        ${HIPCC_INCLUDE_FLAGS}
        -DAEGISBIT_HAS_GPU=1
        -std=c++17
        -fPIC
        -fno-rtti
        -o ${HSA_POOL_GPU_OBJ}
        ${HSA_POOL_GPU_SRC}
      DEPENDS ${HSA_POOL_GPU_SRC} GTest::gtest
      COMMENT "Compiling HSAPoolGPUGTest with hipcc (${TEST_GPU_ARCH})"
      VERBATIM
    )

    add_custom_target(hsa_pool_gpu_obj DEPENDS ${HSA_POOL_GPU_OBJ})

    add_executable(hsa_pool_gpu_gtest ${HSA_POOL_GPU_OBJ})
    add_dependencies(hsa_pool_gpu_gtest hsa_pool_gpu_obj)
    set_target_properties(hsa_pool_gpu_gtest PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(hsa_pool_gpu_gtest
      aegisbit
      GTest::gtest_main
    )

    add_test(NAME HSAPoolGPUGTest COMMAND hsa_pool_gpu_gtest)
    set_tests_properties(HSAPoolGPUGTest PROPERTIES
      ENVIRONMENT "GTEST_COLOR=1"
      LABELS "gpu"
    )

    #---------------------------------------------------------------------------
    # Above-The-Count GPU Test
    # Patches a real GEMM kernel using above-the-count registers (no liveness
    # analysis), loads and dispatches on GPU, compares results.
    #---------------------------------------------------------------------------
    set(ATC_GPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/AboveTheCountGPUTest.cpp")
    set(ATC_GPU_OBJ "${CMAKE_CURRENT_BINARY_DIR}/AboveTheCountGPUTest.o")

    add_custom_command(
      OUTPUT ${ATC_GPU_OBJ}
      COMMAND ${HIPCC_EXECUTABLE}
        -c
        --offload-arch=${TEST_GPU_ARCH}
        ${HIPCC_INCLUDE_FLAGS}
        -I${CMAKE_SOURCE_DIR}/test/unit
        -DAEGISBIT_HAS_GPU=1
        -std=c++17
        -fPIC
        -fno-rtti
        -o ${ATC_GPU_OBJ}
        ${ATC_GPU_SRC}
      DEPENDS ${ATC_GPU_SRC}
      COMMENT "Compiling AboveTheCountGPUTest with hipcc"
      VERBATIM
    )

    add_custom_target(atc_gpu_obj DEPENDS ${ATC_GPU_OBJ})

    add_executable(above_the_count_gpu_test ${ATC_GPU_OBJ})
    add_dependencies(above_the_count_gpu_test atc_gpu_obj)
    set_target_properties(above_the_count_gpu_test PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(above_the_count_gpu_test
      aegisbit
    )

    add_test(NAME AboveTheCountGPUTest COMMAND above_the_count_gpu_test)
    set_tests_properties(AboveTheCountGPUTest PROPERTIES
      ENVIRONMENT "GTEST_COLOR=1"
      LABELS "gpu;above-the-count"
      TIMEOUT 30
    )

    #---------------------------------------------------------------------------
    # Instrumented Trampoline GPU Test
    # Per-lane address capture to trace buffer for coalescing analysis.
    #---------------------------------------------------------------------------
    set(INST_GPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/InstrumentedGPUTest.cpp")
    set(INST_GPU_OBJ "${CMAKE_CURRENT_BINARY_DIR}/InstrumentedGPUTest.o")

    add_custom_command(
      OUTPUT ${INST_GPU_OBJ}
      COMMAND ${HIPCC_EXECUTABLE}
        -c
        --offload-arch=${TEST_GPU_ARCH}
        ${HIPCC_INCLUDE_FLAGS}
        -I${CMAKE_SOURCE_DIR}/test/unit
        -DAEGISBIT_HAS_GPU=1
        -std=c++17
        -fPIC
        -fno-rtti
        -o ${INST_GPU_OBJ}
        ${INST_GPU_SRC}
      DEPENDS ${INST_GPU_SRC}
      COMMENT "Compiling InstrumentedGPUTest with hipcc"
      VERBATIM
    )

    add_custom_target(inst_gpu_obj DEPENDS ${INST_GPU_OBJ})

    add_executable(instrumented_gpu_test ${INST_GPU_OBJ})
    add_dependencies(instrumented_gpu_test inst_gpu_obj)
    set_target_properties(instrumented_gpu_test PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(instrumented_gpu_test
      aegisbit
    )

    add_test(NAME InstrumentedGPUTest COMMAND instrumented_gpu_test)
    set_tests_properties(InstrumentedGPUTest PROPERTIES
      ENVIRONMENT "GTEST_COLOR=1"
      LABELS "gpu;instrumented;profiler"
      TIMEOUT 60
    )

    #---------------------------------------------------------------------------
    # Huge-memory stress kernel diagnostic GTest
    # Runs the exact stress_huge_mem kernel and prints launch diagnostics.
    #---------------------------------------------------------------------------
    set(HUGE_MEM_GPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/HugeMemKernelGTest.cpp")
    set(HUGE_MEM_GPU_OBJ "${CMAKE_CURRENT_BINARY_DIR}/HugeMemKernelGTest.o")

    add_custom_command(
      OUTPUT ${HUGE_MEM_GPU_OBJ}
      COMMAND ${HIPCC_EXECUTABLE}
        -c
        --offload-arch=${TEST_GPU_ARCH}
        ${HIPCC_INCLUDE_FLAGS}
        -DAEGISBIT_HAS_GPU=1
        -std=c++17
        -fPIC
        -fno-rtti
        -o ${HUGE_MEM_GPU_OBJ}
        ${HUGE_MEM_GPU_SRC}
      DEPENDS ${HUGE_MEM_GPU_SRC} GTest::gtest
      COMMENT "Compiling HugeMemKernelGTest with hipcc (${TEST_GPU_ARCH})"
      VERBATIM
    )

    add_custom_target(huge_mem_gpu_obj DEPENDS ${HUGE_MEM_GPU_OBJ})

    add_executable(huge_mem_kernel_gtest ${HUGE_MEM_GPU_OBJ})
    add_dependencies(huge_mem_kernel_gtest huge_mem_gpu_obj)
    set_target_properties(huge_mem_kernel_gtest PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(huge_mem_kernel_gtest
      aegisbit
      GTest::gtest_main
    )

    add_test(NAME HugeMemKernelGTest COMMAND huge_mem_kernel_gtest)
    set_tests_properties(HugeMemKernelGTest PROPERTIES
      ENVIRONMENT "GTEST_COLOR=1"
      LABELS "gpu;stress;diagnostic"
      TIMEOUT 120
    )

    #---------------------------------------------------------------------------
    # SwapPC GPU Dispatch Regression Test
    # Patches mega_gather with both SharedBody and SwapPC, dispatches on GPU.
    #---------------------------------------------------------------------------
    set(SWAPPC_GPU_SRC "${CMAKE_CURRENT_SOURCE_DIR}/SwapPCDispatchGPUTest.cpp")
    set(SWAPPC_GPU_OBJ "${CMAKE_CURRENT_BINARY_DIR}/SwapPCDispatchGPUTest.o")

    add_custom_command(
      OUTPUT ${SWAPPC_GPU_OBJ}
      COMMAND ${HIPCC_EXECUTABLE}
        -c
        --offload-arch=${TEST_GPU_ARCH}
        ${HIPCC_INCLUDE_FLAGS}
        -DAEGISBIT_HAS_GPU=1
        -DMEGA_GATHER_FIXTURE_PATH="${CMAKE_SOURCE_DIR}/test/unit/fixtures/mega_gather_gfx950.bin"
        -std=c++17
        -fPIC
        -fno-rtti
        -o ${SWAPPC_GPU_OBJ}
        ${SWAPPC_GPU_SRC}
      DEPENDS ${SWAPPC_GPU_SRC}
      COMMENT "Compiling SwapPCDispatchGPUTest with hipcc (${TEST_GPU_ARCH})"
      VERBATIM
    )

    add_custom_target(swappc_gpu_obj DEPENDS ${SWAPPC_GPU_OBJ})

    add_executable(swappc_dispatch_gpu_test ${SWAPPC_GPU_OBJ})
    add_dependencies(swappc_dispatch_gpu_test swappc_gpu_obj)
    set_target_properties(swappc_dispatch_gpu_test PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(swappc_dispatch_gpu_test
      aegisbit
    )

    add_test(NAME SwapPCDispatchGPUTest COMMAND swappc_dispatch_gpu_test)
    set_tests_properties(SwapPCDispatchGPUTest PROPERTIES
      ENVIRONMENT "GTEST_COLOR=1"
      LABELS "gpu;swappc;regression"
      TIMEOUT 120
    )

    #---------------------------------------------------------------------------
    # ZeroSGPR + ScratchSpill GPU Regression Test
    # Patches flash_attn kernel (480 VGPRs, 112 SGPRs) which forces ZeroSGPR
    # mode with scratch spill offset > 4095.
    #---------------------------------------------------------------------------
    set(ZERO_SGPR_SPILL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/ZeroSGPRSpillGPUTest.cpp")
    set(ZERO_SGPR_SPILL_OBJ "${CMAKE_CURRENT_BINARY_DIR}/ZeroSGPRSpillGPUTest.o")

    add_custom_command(
      OUTPUT ${ZERO_SGPR_SPILL_OBJ}
      COMMAND ${HIPCC_EXECUTABLE}
        -c
        --offload-arch=${TEST_GPU_ARCH}
        ${HIPCC_INCLUDE_FLAGS}
        -DAEGISBIT_HAS_GPU=1
        -DFLASH_ATTN_FIXTURE_PATH="${CMAKE_SOURCE_DIR}/test/unit/fixtures/flash_attn_gfx950.bin"
        -std=c++17
        -fPIC
        -fno-rtti
        -o ${ZERO_SGPR_SPILL_OBJ}
        ${ZERO_SGPR_SPILL_SRC}
      DEPENDS ${ZERO_SGPR_SPILL_SRC}
      COMMENT "Compiling ZeroSGPRSpillGPUTest with hipcc (${TEST_GPU_ARCH})"
      VERBATIM
    )

    add_custom_target(zero_sgpr_spill_gpu_obj DEPENDS ${ZERO_SGPR_SPILL_OBJ})

    add_executable(zero_sgpr_spill_gpu_test ${ZERO_SGPR_SPILL_OBJ})
    add_dependencies(zero_sgpr_spill_gpu_test zero_sgpr_spill_gpu_obj)
    set_target_properties(zero_sgpr_spill_gpu_test PROPERTIES LINKER_LANGUAGE CXX)
    target_link_libraries(zero_sgpr_spill_gpu_test
      aegisbit
    )

    add_test(NAME ZeroSGPRSpillGPUTest COMMAND zero_sgpr_spill_gpu_test)
    set_tests_properties(ZeroSGPRSpillGPUTest PROPERTIES
      ENVIRONMENT "GTEST_COLOR=1"
      LABELS "gpu;zero-sgpr;regression"
      TIMEOUT 120
    )

  else()
    message(STATUS "hipcc not found, skipping GPU integration tests")
  endif()

endif()
