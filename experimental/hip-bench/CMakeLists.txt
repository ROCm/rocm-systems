cmake_minimum_required(VERSION 3.21)

# Ahead of project() on purpose. CMAKE_BUILD_TYPE is empty here unless the user
# named one on the command line, whereas project() on Windows seeds it to Debug
# before we get a say, and a Debug build would swamp what this measures.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Build type" FORCE)
endif()

# The ROCm prefix is on the default search path on neither platform: the Windows
# installer exports ROCM_PATH/HIP_PATH, and /opt/rocm is the Linux convention.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH)
    elseif(DEFINED ENV{HIP_PATH})
        file(TO_CMAKE_PATH "$ENV{HIP_PATH}" ROCM_PATH)
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()

# On Windows the HIP compiler is ROCm's clang driven GNU-style, and CMake refuses
# to mix that with MSVC for C++. Every translation unit here is compiled as HIP
# anyway, so default C++ to the same compiler rather than cl.
if(WIN32
   AND NOT DEFINED CMAKE_CXX_COMPILER
   AND EXISTS "${ROCM_PATH}/lib/llvm/bin/clang++.exe")
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++.exe")
endif()

project(hip-bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find HIP. We need it because the .hip translation unit must be compiled
# with hipcc and linked against hip::host.
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
find_package(hip REQUIRED CONFIG)

# The Windows ROCm layout puts the device bitcode under lib/llvm/amdgcn/bitcode,
# which is not one of the locations clang derives from --rocm-path, so name it.
# This has to happen before enable_language(HIP): compiler identification itself
# compiles a device translation unit and fails without the bitcode.
if(WIN32
   AND NOT DEFINED CMAKE_HIP_FLAGS
   AND EXISTS "${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
    set(CMAKE_HIP_FLAGS "--rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode")
endif()

# CMake derives the default architecture from rocm_agent_enumerator, which the
# Windows ROCm package does not ship. Without it the compiler's own fallback
# (gfx906) wins silently and the binary will not run. amdgpu-arch answers the
# same question and is present in both layouts.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES AND NOT EXISTS "${ROCM_PATH}/bin/rocm_agent_enumerator")
    find_program(
        HIPBENCH_AMDGPU_ARCH
        NAMES amdgpu-arch
        HINTS "${ROCM_PATH}/lib/llvm/bin" "${ROCM_PATH}/llvm/bin")
    if(HIPBENCH_AMDGPU_ARCH)
        execute_process(
            COMMAND "${HIPBENCH_AMDGPU_ARCH}"
            RESULT_VARIABLE hipbench_arch_result
            OUTPUT_VARIABLE hipbench_arch_output
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(hipbench_arch_result EQUAL 0 AND hipbench_arch_output)
            string(REPLACE "\n" ";" hipbench_archs "${hipbench_arch_output}")
            list(REMOVE_DUPLICATES hipbench_archs)
            set(CMAKE_HIP_ARCHITECTURES
                "${hipbench_archs}"
                CACHE STRING "HIP architectures")
        endif()
    endif()
endif()

enable_language(HIP)

add_executable(hipbench src/main.cpp src/kernels.hip)

set_source_files_properties(src/kernels.hip src/main.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipbench PRIVATE hip::host)
target_compile_options(hipbench PRIVATE -Wall -Wextra)
