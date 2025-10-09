cmake_minimum_required(VERSION 3.10)
project(hip_ebpf_trace VERSION 1.0.0 LANGUAGES C CXX)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2 -g")
set(CMAKE_C_FLAGS_DEBUG "-DDEBUG -g3")
set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -O3")

# Find required packages
find_package(PkgConfig REQUIRED)

# Check for libbpf
pkg_check_modules(LIBBPF REQUIRED libbpf)
if(NOT LIBBPF_FOUND)
    message(FATAL_ERROR "libbpf not found. Install with: sudo apt install libbpf-dev")
endif()

# Check for libelf
find_library(ELF_LIB elf REQUIRED)
if(NOT ELF_LIB)
    message(FATAL_ERROR "libelf not found. Install with: sudo apt install libelf-dev")
endif()

# Check for zlib
find_library(Z_LIB z REQUIRED)
if(NOT Z_LIB)
    message(FATAL_ERROR "zlib not found. Install with: sudo apt install zlib1g-dev")
endif()

# Chrome tracing JSON format is used for Perfetto output
message(STATUS "Chrome tracing JSON format enabled for Perfetto output")

# Check for required tools
find_program(CLANG clang REQUIRED)
find_program(BPFTOOL bpftool REQUIRED)
find_program(PYTHON3 python3 REQUIRED)

if(NOT CLANG)
    message(FATAL_ERROR "clang not found. Install with: sudo apt install clang")
endif()

if(NOT BPFTOOL)
    message(FATAL_ERROR "bpftool not found. Install with: sudo apt install linux-tools-common linux-tools-generic")
endif()

if(NOT PYTHON3)
    message(FATAL_ERROR "python3 not found. Install with: sudo apt install python3")
endif()

# Include directories
include_directories(${LIBBPF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# eBPF source files
set(EBPF_SOURCES
    hip_trace.bpf.c
)

# C source files
set(C_SOURCES
    hip_trace.c
    chrome_trace_writer.c
    ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.c
)

# eBPF object file
set(EBPF_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/hip_trace.bpf.o)
set(SKELETON_HEADER ${CMAKE_CURRENT_BINARY_DIR}/hip_trace.skel.h)

# Find HIP runtime header
find_path(HIP_INCLUDE_DIR hip_runtime_api.h
    PATHS /opt/rocm/include/hip
          /usr/local/rocm/include/hip
          /usr/include/hip
    NO_DEFAULT_PATH
)

if(NOT HIP_INCLUDE_DIR)
    message(WARNING "HIP runtime header not found. Using fallback function list.")
    set(HIP_HEADER_FILE "")
else()
    set(HIP_HEADER_FILE ${HIP_INCLUDE_DIR}/hip_runtime_api.h)
    message(STATUS "Found HIP runtime header: ${HIP_HEADER_FILE}")
endif()

# Custom command to generate HIP functions code from headers
if(HIP_HEADER_FILE)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.c
        COMMAND ${PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/generate_hip_functions_from_headers.py ${HIP_HEADER_FILE} ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/generate_hip_functions_from_headers.py ${HIP_HEADER_FILE}
        COMMENT "Generating HIP functions code from headers"
        VERBATIM
    )
else()
    # Fallback: create empty files if header not found
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.c
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.c
        COMMENT "Creating empty HIP functions files (header not found)"
        VERBATIM
    )
endif()

# Custom command to compile eBPF program
add_custom_command(
    OUTPUT ${EBPF_OBJECT}
    COMMAND ${CLANG} -O2 -g -target bpf -c ${CMAKE_CURRENT_SOURCE_DIR}/hip_trace.bpf.c -o ${EBPF_OBJECT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hip_trace.bpf.c
    COMMENT "Compiling eBPF program"
    VERBATIM
)

# Custom target for eBPF object
add_custom_target(ebpf_object
    DEPENDS ${EBPF_OBJECT}
    COMMENT "eBPF object file"
)

# Custom command to generate skeleton header
add_custom_command(
    OUTPUT ${SKELETON_HEADER}
    COMMAND ${BPFTOOL} gen skeleton ${EBPF_OBJECT} > ${SKELETON_HEADER}
    DEPENDS ${EBPF_OBJECT}
    COMMENT "Generating eBPF skeleton header"
    VERBATIM
)

# Custom target for skeleton header
add_custom_target(skeleton_header
    DEPENDS ${SKELETON_HEADER}
    COMMENT "eBPF skeleton header"
)

# Custom target for generated HIP functions
add_custom_target(generated_hip_functions
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.bpf.c ${CMAKE_CURRENT_BINARY_DIR}/hip_functions_generated.c
    COMMENT "Generated HIP functions"
)

# Main executable
add_executable(hip_trace ${C_SOURCES})
add_dependencies(hip_trace skeleton_header)
add_dependencies(hip_trace generated_hip_functions)

# Link libraries
target_link_libraries(hip_trace
    ${LIBBPF_LIBRARIES}
    ${ELF_LIB}
    ${Z_LIB}
)

# No additional libraries needed for Chrome tracing JSON format

# Set target properties
set_target_properties(hip_trace PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

# Install targets
install(TARGETS hip_trace
    RUNTIME DESTINATION bin
)

# HSA Queue Interceptor (LD_PRELOAD-based)
# Find HSA runtime
find_path(HSA_INCLUDE_DIR hsa/hsa.h
    PATHS /opt/rocm/include
          /usr/local/rocm/include
          /usr/include
    NO_DEFAULT_PATH
)

if(NOT HSA_INCLUDE_DIR)
    message(WARNING "HSA runtime header not found. Skipping HSA queue tools.")
else()
    message(STATUS "Found HSA runtime header: ${HSA_INCLUDE_DIR}")

    # Find ROCprofiler SDK
    find_path(ROCPROFILER_INCLUDE_DIR rocprofiler-sdk/registration.h
        PATHS /opt/rocm/include
              /usr/local/rocm/include
              /usr/include
        NO_DEFAULT_PATH
    )

    find_library(ROCPROFILER_LIBRARY rocprofiler-sdk
        PATHS /opt/rocm/lib
              /usr/local/rocm/lib
              /usr/lib
        NO_DEFAULT_PATH
    )

    if(NOT ROCPROFILER_INCLUDE_DIR OR NOT ROCPROFILER_LIBRARY)
        message(WARNING "ROCprofiler SDK not found. HSA Queue Interceptor will have limited functionality.")
        set(ROCPROFILER_INCLUDE_DIR "")
        set(ROCPROFILER_LIBRARY "")
    else()
        message(STATUS "Found ROCprofiler SDK: ${ROCPROFILER_LIBRARY}")
        message(STATUS "  Include: ${ROCPROFILER_INCLUDE_DIR}")
    endif()

    # Detect architecture for eBPF compilation
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH_UNAME
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(ARCH_UNAME STREQUAL "x86_64")
        set(BPF_ARCH "x86")
        set(BPF_TARGET_ARCH "__TARGET_ARCH_x86")
    elseif(ARCH_UNAME STREQUAL "aarch64")
        set(BPF_ARCH "arm64")
        set(BPF_TARGET_ARCH "__TARGET_ARCH_arm64")
    else()
        message(FATAL_ERROR "Unsupported architecture: ${ARCH_UNAME}")
    endif()

    message(STATUS "eBPF target architecture: ${BPF_ARCH} (${BPF_TARGET_ARCH})")

    # NOTE: Old HSA Queue Tracer has been replaced by the Unified GPU Tracer below

    # HSA Hybrid Shim (ROCprofiler SDK registration with LD_PRELOAD)
    # NOTE: This now sends kernel events to eBPF, no longer writes Chrome Trace directly
    add_library(hsa_hybrid_shim SHARED
        hsa_hybrid_shim.cpp
    )

    target_include_directories(hsa_hybrid_shim PRIVATE
        ${HSA_INCLUDE_DIR}
        ${ROCPROFILER_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBBPF_INCLUDE_DIRS}
    )

    target_link_libraries(hsa_hybrid_shim
        ${CMAKE_DL_LIBS}
        pthread
        ${LIBBPF_LIBRARIES}
        ${ELF_LIB}
        ${ROCPROFILER_LIBRARY}
    )

    set_target_properties(hsa_hybrid_shim PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Install hybrid shim
    install(TARGETS hsa_hybrid_shim
        LIBRARY DESTINATION lib
    )

    message(STATUS "")
    message(STATUS "=== Unified HIP + Kernel Tracer (NO rocprofiler-sdk for HIP API tracing) ===")

    # Unified eBPF program (HIP uprobes + kernel events from shim)
    set(UNIFIED_HIP_KERNEL_EBPF_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_unified.bpf.o)
    set(UNIFIED_HIP_KERNEL_SKELETON_HEADER ${CMAKE_CURRENT_BINARY_DIR}/hip_kernel_unified.skel.h)

    add_custom_command(
        OUTPUT ${UNIFIED_HIP_KERNEL_EBPF_OBJECT}
        COMMAND ${CLANG} -O2 -g -target bpf
                -D${BPF_TARGET_ARCH}
                -I${CMAKE_CURRENT_SOURCE_DIR}
                -I/usr/include
                -c ${CMAKE_CURRENT_SOURCE_DIR}/hip_kernel_unified.bpf.c
                -o ${UNIFIED_HIP_KERNEL_EBPF_OBJECT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/hip_kernel_unified.bpf.c
        COMMENT "Compiling Unified HIP + Kernel eBPF program (${BPF_ARCH})"
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${UNIFIED_HIP_KERNEL_SKELETON_HEADER}
        COMMAND ${BPFTOOL} gen skeleton ${UNIFIED_HIP_KERNEL_EBPF_OBJECT} > ${UNIFIED_HIP_KERNEL_SKELETON_HEADER}
        DEPENDS ${UNIFIED_HIP_KERNEL_EBPF_OBJECT}
        COMMENT "Generating Unified HIP + Kernel skeleton header"
        VERBATIM
    )

    add_custom_target(unified_hip_kernel_skeleton
        DEPENDS ${UNIFIED_HIP_KERNEL_SKELETON_HEADER}
        COMMENT "Unified HIP + Kernel skeleton header"
    )

    # Unified tracer executable
    add_executable(hip_kernel_unified_tracer
        hip_kernel_unified_tracer.c
        chrome_trace_writer.c
    )

    add_dependencies(hip_kernel_unified_tracer unified_hip_kernel_skeleton)

    target_include_directories(hip_kernel_unified_tracer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${LIBBPF_INCLUDE_DIRS}
    )

    target_link_libraries(hip_kernel_unified_tracer
        ${LIBBPF_LIBRARIES}
        ${ELF_LIB}
        ${Z_LIB}
    )

    set_target_properties(hip_kernel_unified_tracer PROPERTIES
        C_STANDARD 99
        C_STANDARD_REQUIRED ON
    )

    install(TARGETS hip_kernel_unified_tracer
        RUNTIME DESTINATION bin
    )

    message(STATUS "")
    message(STATUS "=== Unified HIP + Kernel Tracer (Pure eBPF + Lightweight Shim) ===")
    message(STATUS "Out-of-process tool: hip_kernel_unified_tracer")
    message(STATUS "  Usage: sudo ./hip_kernel_unified_tracer -l /opt/rocm/lib/libamdhip64.so -o trace.json")
    message(STATUS "Lightweight shim: libhsa_hybrid_shim.so")
    message(STATUS "  Usage: LD_PRELOAD=./libhsa_hybrid_shim.so <app>")
    message(STATUS "Features:")
    message(STATUS "  - HIP API tracing via eBPF uprobes (NO rocprofiler-sdk)")
    message(STATUS "  - Kernel dispatch tracing via lightweight shim (uses rocprofiler-sdk ONLY for HSA table access)")
    message(STATUS "  - GPU-side kernel timestamps via HSA profiling API")
    message(STATUS "  - Unified Chrome Trace JSON output")
endif()

# Install scripts
install(PROGRAMS
    test_csv_output.sh
    demo_csv_analysis.sh
    test_queue_tracer.sh
    trace_hip_and_kernels.sh
    DESTINATION bin
)

# Install documentation
install(FILES
    README.md
    eBPF_HIP_Tracing_Technical_Guide.md
    DESTINATION share/doc/hip_ebpf_trace
)

# Custom targets for development
add_custom_target(clean-ebpf
    COMMAND ${CMAKE_COMMAND} -E remove -f ${EBPF_OBJECT} ${SKELETON_HEADER}
    COMMENT "Cleaning eBPF generated files"
)

add_custom_target(rebuild-ebpf
    COMMAND ${CMAKE_COMMAND} -E remove -f ${EBPF_OBJECT} ${SKELETON_HEADER}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target hip_trace
    COMMENT "Rebuilding eBPF program"
)

# Print build information
message(STATUS "Build configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message(STATUS "  CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "  LIBBPF_INCLUDE_DIRS: ${LIBBPF_INCLUDE_DIRS}")
message(STATUS "  LIBBPF_LIBRARIES: ${LIBBPF_LIBRARIES}")
message(STATUS "  ELF_LIB: ${ELF_LIB}")
message(STATUS "  Z_LIB: ${Z_LIB}")

# Check for vmlinux.h
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/vmlinux.h")
    message(WARNING "vmlinux.h not found. Generate with: bpftool btf dump file /sys/kernel/btf/vmlinux format c > vmlinux.h")
endif()

# Print usage instructions
message(STATUS "")
message(STATUS "Build completed successfully!")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  sudo ./hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.csv")
message(STATUS "  ./demo_csv_analysis.sh trace.csv")
message(STATUS "")
message(STATUS "Note: Root privileges required for eBPF program loading")
