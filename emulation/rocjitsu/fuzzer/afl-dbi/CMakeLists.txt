# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(rj_configure_target)

set(ROCFUZZ_ROCM_PATH
    "/opt/rocm"
    CACHE PATH
    "ROCm installation used by RocFuzz"
)

find_library(
    ROCFUZZ_HIP_RUNTIME
    amdhip64
    HINTS "${ROCFUZZ_ROCM_PATH}"
    ENV ROCM_PATH
    PATH_SUFFIXES lib lib64
)
find_path(
    ROCFUZZ_HIP_INCLUDE_DIR
    hip/hip_runtime_api.h
    HINTS "${ROCFUZZ_ROCM_PATH}"
    ENV ROCM_PATH
    PATH_SUFFIXES include
)

add_executable(
    rocjitsu_afl_amdgpu_elf_reader_test
    tests/amdgpu_elf_reader_test.cpp
)
target_link_libraries(
    rocjitsu_afl_amdgpu_elf_reader_test
    PRIVATE
        rocjitsu_analysis
        rocjitsu_code
        rocjitsu_isa
        rocjitsu_isa_cdna1
        rocjitsu_isa_cdna2
        rocjitsu_isa_cdna3
        rocjitsu_isa_cdna4
        rocjitsu_isa_rdna1
        rocjitsu_isa_rdna2
        rocjitsu_isa_rdna3
        rocjitsu_isa_rdna3_5
        rocjitsu_isa_rdna4
        rocjitsu_isa_risc_v
        simdojo
        rocjitsu_vm
        rocjitsu_vm_amdgpu
        rocjitsu_vm_risc_v
        rocjitsu_config
        util
        flatbuffers
        Threads::Threads
        GTest::gtest_main
)
rj_configure_target(rocjitsu_afl_amdgpu_elf_reader_test TEST)
add_test(
    NAME RocjitsuAflDbi.AmdGpuElfReader
    COMMAND rocjitsu_afl_amdgpu_elf_reader_test
)

if(ROCFUZZ_HIP_RUNTIME AND ROCFUZZ_HIP_INCLUDE_DIR)
    add_library(rocjitsu_afl_preload SHARED runtime/afl_preload.cpp)
    target_include_directories(
        rocjitsu_afl_preload
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(
        rocjitsu_afl_preload
        SYSTEM
        PRIVATE ${ROCFUZZ_HIP_INCLUDE_DIR}
    )
    target_link_libraries(
        rocjitsu_afl_preload
        PRIVATE
            rocjitsu_analysis
            rocjitsu_code
            rocjitsu_isa
            rocjitsu_isa_cdna1
            rocjitsu_isa_cdna2
            rocjitsu_isa_cdna3
            rocjitsu_isa_cdna4
            rocjitsu_isa_rdna1
            rocjitsu_isa_rdna2
            rocjitsu_isa_rdna3
            rocjitsu_isa_rdna3_5
            rocjitsu_isa_rdna4
            rocjitsu_isa_risc_v
            simdojo
            rocjitsu_vm
            rocjitsu_vm_amdgpu
            rocjitsu_vm_risc_v
            rocjitsu_config
            util
            flatbuffers
            Threads::Threads
            ${ROCFUZZ_HIP_RUNTIME}
            ${CMAKE_DL_LIBS}
    )
    target_compile_definitions(
        rocjitsu_afl_preload
        PRIVATE __HIP_PLATFORM_AMD__
    )
    rj_configure_target(rocjitsu_afl_preload PRIVATE_INCLUDES)
    set_target_properties(
        rocjitsu_afl_preload
        PROPERTIES OUTPUT_NAME rocjitsu_afl_preload
    )
    add_executable(
        rocjitsu_afl_preload_symbols_test
        tests/afl_preload_symbols_test.cpp
    )
    target_compile_definitions(
        rocjitsu_afl_preload_symbols_test
        PRIVATE ROCJITSU_AFL_PRELOAD_PATH="$<TARGET_FILE:rocjitsu_afl_preload>"
    )
    target_link_libraries(
        rocjitsu_afl_preload_symbols_test
        PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS}
    )
    rj_configure_target(rocjitsu_afl_preload_symbols_test TEST)
    add_dependencies(rocjitsu_afl_preload_symbols_test rocjitsu_afl_preload)
    add_test(
        NAME RocjitsuAflDbi.PreloadSymbols
        COMMAND rocjitsu_afl_preload_symbols_test
    )

    add_library(rocjitsu_afl_fake_hip_runtime SHARED tests/fake_hip_runtime.cpp)
    rj_configure_target(rocjitsu_afl_fake_hip_runtime PRIVATE_INCLUDES)

    add_library(rocjitsu_afl_fake_hsa_runtime SHARED tests/fake_hsa_runtime.cpp)
    target_include_directories(
        rocjitsu_afl_fake_hsa_runtime
        SYSTEM
        PRIVATE ${ROCFUZZ_HIP_INCLUDE_DIR}
    )
    rj_configure_target(rocjitsu_afl_fake_hsa_runtime PRIVATE_INCLUDES)

    add_executable(
        rocjitsu_afl_preload_runtime_test
        tests/afl_preload_runtime_test.cpp
    )
    target_include_directories(
        rocjitsu_afl_preload_runtime_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_definitions(
        rocjitsu_afl_preload_runtime_test
        PRIVATE
            ROCJITSU_AFL_PRELOAD_PATH="$<TARGET_FILE:rocjitsu_afl_preload>"
            ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH="$<TARGET_FILE:rocjitsu_afl_fake_hip_runtime>"
    )
    target_link_libraries(
        rocjitsu_afl_preload_runtime_test
        PRIVATE GTest::gtest_main ${CMAKE_DL_LIBS}
    )
    rj_configure_target(rocjitsu_afl_preload_runtime_test TEST)
    add_dependencies(
        rocjitsu_afl_preload_runtime_test
        rocjitsu_afl_preload
        rocjitsu_afl_fake_hip_runtime
    )
    add_test(
        NAME RocjitsuAflDbi.PreloadRuntime
        COMMAND rocjitsu_afl_preload_runtime_test
    )

    add_executable(
        rocjitsu_afl_preload_hsa_reader_test
        tests/afl_preload_hsa_reader_test.cpp
    )
    target_include_directories(
        rocjitsu_afl_preload_hsa_reader_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_include_directories(
        rocjitsu_afl_preload_hsa_reader_test
        SYSTEM
        PRIVATE ${ROCFUZZ_HIP_INCLUDE_DIR}
    )
    target_compile_definitions(
        rocjitsu_afl_preload_hsa_reader_test
        PRIVATE
            ROCJITSU_AFL_PRELOAD_PATH="$<TARGET_FILE:rocjitsu_afl_preload>"
            ROCJITSU_AFL_FAKE_HIP_RUNTIME_PATH="$<TARGET_FILE:rocjitsu_afl_fake_hip_runtime>"
            ROCJITSU_AFL_FAKE_HSA_RUNTIME_PATH="$<TARGET_FILE:rocjitsu_afl_fake_hsa_runtime>"
    )
    target_link_libraries(
        rocjitsu_afl_preload_hsa_reader_test
        PRIVATE
            rocjitsu_analysis
            rocjitsu_code
            rocjitsu_isa
            rocjitsu_isa_cdna1
            rocjitsu_isa_cdna2
            rocjitsu_isa_cdna3
            rocjitsu_isa_cdna4
            rocjitsu_isa_rdna1
            rocjitsu_isa_rdna2
            rocjitsu_isa_rdna3
            rocjitsu_isa_rdna3_5
            rocjitsu_isa_rdna4
            rocjitsu_isa_risc_v
            simdojo
            rocjitsu_vm
            rocjitsu_vm_amdgpu
            rocjitsu_vm_risc_v
            rocjitsu_config
            util
            flatbuffers
            Threads::Threads
            GTest::gtest_main
            ${CMAKE_DL_LIBS}
    )
    rj_configure_target(rocjitsu_afl_preload_hsa_reader_test TEST)
    add_dependencies(
        rocjitsu_afl_preload_hsa_reader_test
        rocjitsu_afl_preload
        rocjitsu_afl_fake_hip_runtime
        rocjitsu_afl_fake_hsa_runtime
    )
    add_test(
        NAME RocjitsuAflDbi.PreloadHsaReader
        COMMAND rocjitsu_afl_preload_hsa_reader_test
    )
else()
    message(
        STATUS
        "RocFuzz AFL preload disabled (amdhip64 or HIP headers not found)"
    )
endif()
