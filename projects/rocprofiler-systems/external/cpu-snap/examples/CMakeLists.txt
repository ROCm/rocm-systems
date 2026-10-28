add_subdirectory(common)

# Sampling examples: capture via a core::memory_sink, decode with
# core::sample_reconstructor, and post-process after recording stops.
#
# Every target here builds on cpu_snap::sampling (sampler, autoload) and
# cpu_snap::processing (symbolizer, slice emitter).
if(NOT CPU_SNAP_BUILD_SAMPLING OR NOT CPU_SNAP_BUILD_PROCESSING)
    message(STATUS
        "cpu_snap examples: skipped (need CPU_SNAP_BUILD_SAMPLING and CPU_SNAP_BUILD_PROCESSING)")
    return()
endif()

# Minimal single-thread sampling-only demo: smallest viable usage of
# sampling::sampler + perfetto writer for copy-paste reference.
add_executable(single_thread_callstack_demo single_thread_callstack_demo.cpp)
target_link_libraries(single_thread_callstack_demo PRIVATE cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo)
target_compile_options(single_thread_callstack_demo PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -no-pie)
target_link_options(single_thread_callstack_demo PRIVATE -no-pie)

add_executable(multi_thread_sample_demo multi_thread_sample_demo.cpp)
target_link_libraries(multi_thread_sample_demo PRIVATE cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo)
target_compile_options(multi_thread_sample_demo PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -no-pie)
target_link_options(multi_thread_sample_demo PRIVATE -no-pie)

# Symbolizer inline-frame expansion demo. Built at -O2 so the
# always_inline helpers actually fold into one physical frame; the
# symbolizer surfaces the chain back out via addr2line -i.
add_executable(inline_frames_demo inline_frames_demo.cpp)
target_link_libraries(inline_frames_demo PRIVATE cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo)
target_compile_options(inline_frames_demo PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -no-pie)
target_link_options(inline_frames_demo PRIVATE -no-pie)

# Runs the same workload under both backends, emits two traces, prints
# a comparison table (samples, lost, duration_ms).
add_executable(backend_comparison_demo backend_comparison_demo.cpp)
target_link_libraries(backend_comparison_demo PRIVATE cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo)
target_compile_options(backend_comparison_demo PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -no-pie)
target_link_options(backend_comparison_demo PRIVATE -no-pie)

# 8-frame, non-inlined call chain where each frame does a distinctly
# different amount of its own work, so a capture visibly shows
# self/inclusive time diverging by depth. See file header for the chain.
add_executable(deep_chain_callstack_demo deep_chain_callstack_demo.cpp)
target_link_libraries(deep_chain_callstack_demo PRIVATE
    cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo)
target_compile_options(deep_chain_callstack_demo PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -no-pie)
target_link_options(deep_chain_callstack_demo PRIVATE -no-pie)

add_executable(multi_thread_callstack_demo multi_thread_callstack_demo.cpp)
target_link_libraries(multi_thread_callstack_demo PRIVATE
    cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo)
target_compile_options(multi_thread_callstack_demo PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -no-pie)
target_link_options(multi_thread_callstack_demo PRIVATE -no-pie)

# LD_PRELOAD autoload sampler
add_library(callstack_autoload SHARED
    autoload/autoload_sampler.cpp)
target_link_libraries(callstack_autoload PRIVATE cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo)
target_compile_options(callstack_autoload PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -fvisibility=hidden)
set_target_properties(callstack_autoload PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

add_executable(sample_target autoload/sample_target.cpp)
target_compile_options(sample_target PRIVATE
    -Wall -Wextra -Wpedantic -O2 -fno-omit-frame-pointer
    -fno-optimize-sibling-calls -g -no-pie)
target_link_options(sample_target PRIVATE -no-pie)
target_link_libraries(sample_target PRIVATE Threads::Threads cpu_snap_workload)

add_custom_target(run_autoload_demo
    COMMAND ${CMAKE_COMMAND} -E env
            "LD_PRELOAD=$<TARGET_FILE:callstack_autoload>"
            "CALLSTACK_TRACE=${CMAKE_BINARY_DIR}/autoload_demo.pftrace"
            $<TARGET_FILE:sample_target>
    DEPENDS callstack_autoload sample_target
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running sample_target under LD_PRELOAD=libcallstack_autoload.so"
    VERBATIM USES_TERMINAL)

# HIP matmul demo: shows host-side CPU stacks running through
# the HIP runtime during a tiled SGEMM (upload/launch/sync/download).
# Sampler bytes are captured to a sink (file + memory) during recording,
# then decoded via cpu_snap::core::sample_reconstructor and symbolized
# in-process after stop(). Opt-in via CPU_SNAP_BUILD_HIP_EXAMPLES since
# most build machines have no ROCm/HIP toolchain installed.
if(CPU_SNAP_BUILD_HIP_EXAMPLES)
    find_program(CPU_SNAP_HIPCC_EXECUTABLE hipcc
        HINTS /home/amd/therock/selected/bin /opt/rocm/bin /opt/rocm-7.2.0/bin)
    if(CPU_SNAP_HIPCC_EXECUTABLE)
        get_filename_component(_hip_bin_dir "${CPU_SNAP_HIPCC_EXECUTABLE}" DIRECTORY)
        get_filename_component(_hip_root_dir "${_hip_bin_dir}" DIRECTORY)

        # CMake's HIP language support refuses the hipcc wrapper script and
        # wants the underlying clang++ driver directly (it passes its own
        # --offload-arch / -x hip flags around it).
        find_program(CPU_SNAP_HIP_CLANGXX_EXECUTABLE clang++
            HINTS "${_hip_root_dir}/llvm/bin" "${_hip_root_dir}/lib/llvm/bin")
        if(NOT CPU_SNAP_HIP_CLANGXX_EXECUTABLE)
            message(FATAL_ERROR
                "hipcc found at ${CPU_SNAP_HIPCC_EXECUTABLE} but no clang++ "
                "next to it; cannot set CMAKE_HIP_COMPILER.")
        endif()

        set(CMAKE_HIP_COMPILER "${CPU_SNAP_HIP_CLANGXX_EXECUTABLE}")
        set(CMAKE_HIP_FLAGS "--rocm-path=${_hip_root_dir}")
        if(NOT CMAKE_HIP_ARCHITECTURES)
            set(CMAKE_HIP_ARCHITECTURES gfx1201)
        endif()
        enable_language(HIP)

        # Optional hipBLAS path: no CMake package config ships with this
        # ROCm/TheRock install, so probe the header/lib directly.
        find_path(CPU_SNAP_HIPBLAS_INCLUDE_DIR hipblas/hipblas.h
            HINTS "${_hip_root_dir}/include")
        find_library(CPU_SNAP_HIPBLAS_LIBRARY hipblas
            HINTS "${_hip_root_dir}/lib")
        if(CPU_SNAP_HIPBLAS_INCLUDE_DIR AND CPU_SNAP_HIPBLAS_LIBRARY)
            set(_have_hipblas 1)
        else()
            set(_have_hipblas 0)
        endif()

        add_executable(hip_matmul_demo hip_matmul_demo.cpp)
        set_source_files_properties(hip_matmul_demo.cpp PROPERTIES LANGUAGE HIP)
        target_include_directories(hip_matmul_demo PRIVATE "${_hip_root_dir}/include")
        target_link_directories(hip_matmul_demo PRIVATE "${_hip_root_dir}/lib")
        target_link_libraries(hip_matmul_demo PRIVATE
            cpu_snap::sampling cpu_snap::core cpu_snap::processing perfetto_writer_demo amdhip64)
        target_compile_definitions(hip_matmul_demo PRIVATE CPU_SNAP_HAVE_HIPBLAS=${_have_hipblas})
        target_compile_options(hip_matmul_demo PRIVATE
            -Wall -Wextra -O2 -fno-omit-frame-pointer
            -fno-optimize-sibling-calls -g -no-pie)
        target_link_options(hip_matmul_demo PRIVATE -no-pie
            "-Wl,-rpath,${_hip_root_dir}/lib")
        if(_have_hipblas)
            target_link_libraries(hip_matmul_demo PRIVATE "${CPU_SNAP_HIPBLAS_LIBRARY}")
            message(STATUS "cpu_snap examples: hip_matmul_demo --blas enabled (hipBLAS=${CPU_SNAP_HIPBLAS_LIBRARY})")
        else()
            message(STATUS "cpu_snap examples: hipBLAS not found; hip_matmul_demo --blas will report unavailable")
        endif()
        message(STATUS "cpu_snap examples: hip_matmul_demo enabled (hipcc=${CPU_SNAP_HIPCC_EXECUTABLE})")
    else()
        message(WARNING
            "CPU_SNAP_BUILD_HIP_EXAMPLES=ON but no hipcc was found; "
            "skipping hip_matmul_demo. Set CPU_SNAP_HIPCC_EXECUTABLE or "
            "add hipcc to PATH to enable it.")
    endif()
endif()
