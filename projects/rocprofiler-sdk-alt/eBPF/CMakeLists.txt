cmake_minimum_required(VERSION 3.10)
project(hip_ebpf_trace VERSION 1.0.0 LANGUAGES C)

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

# Install scripts
install(PROGRAMS
    test_csv_output.sh
    demo_csv_analysis.sh
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
