#
#
#
if(ROCPROFILER_BUILD_CODECOV)
    set(CMAKE_BUILD_TYPE "Coverage")
endif()

#
# by default, activate clang-tidy on all code in the source folder. unittest subfolders
# can add `rocprofiler_deactivate_clang_tidy()` to their CMakeLists.txt to disable
# clang-tidy linting
#
rocprofiler_activate_clang_tidy()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(libexec)
add_subdirectory(bin)
add_subdirectory(docs)
add_subdirectory(share)

# Custom executable for timestamp_ns test (local build, no install)
# add_executable(test_boot test_boot.cpp)

# # Find MPI package for distributed support
find_package(MPI REQUIRED)

# # Find HIP package for ROCm GPU support
find_package(hip REQUIRED)

# # Include directories for source headers and generated files
# target_include_directories(test_boot PRIVATE
#     ${PROJECT_SOURCE_DIR}/source  # Root source dir to resolve relative includes like "lib/common/defines.hpp"
#     ${CMAKE_CURRENT_SOURCE_DIR}/lib/common  # For utility.hpp, defines.hpp
#     ${CMAKE_CURRENT_BINARY_DIR}/include     # For generated rocprofiler-sdk/defines.h
#     ${PROJECT_SOURCE_DIR}/external/glog/src # For glog headers if needed
#     ${MPI_C_INCLUDE_PATH}  # For MPI headers like mpi.h
#     ${HIP_INCLUDE_DIRS}    # For HIP headers like hip_runtime.h
# )

# # Link to common library (utilities like timestamp_ns), MPI, HIP, and pthread
# target_link_libraries(test_boot 
#     rocprofiler-sdk::rocprofiler-sdk-common-library  # Correct target alias for common lib
#     MPI::MPI_CXX  # MPI libraries and flags
#     hip::host     # HIP host runtime for API calls (no device for host-only code)
#     pthread
# )

# # Set C++ standard for test_boot
# set_target_properties(test_boot PROPERTIES 
#     CXX_STANDARD 17 
#     CXX_STANDARD_REQUIRED ON
# )

# New executable for RCCL init test (local build, no install)
add_executable(init_rccl init_rccl.cpp)

# Reuse MPI and HIP (already found for test_boot)

# Find RCCL library explicitly
find_library(RCCL_LIBRARY 
    NAMES rccl librccl 
    PATHS /opt/rocm/lib /opt/rocm-7.0.2/lib  # Standard and versioned ROCm paths
    # NO_DEFAULT_PATH  # Allow system search as fallback
)

if(NOT RCCL_LIBRARY)
    message(FATAL_ERROR "RCCL library not found in /opt/rocm/lib or /opt/rocm-7.0.2/lib. Install rccl-dev or set ROCM_PATH.")
endif()

message(STATUS "Found RCCL library: ${RCCL_LIBRARY}")

# Include directories for MPI and HIP
target_include_directories(init_rccl PRIVATE
    ${MPI_C_INCLUDE_PATH}  # For MPI headers
    ${HIP_INCLUDE_DIRS}    # For HIP headers
)

# Link directories for ROCm libs (to find librccl.so)
target_link_directories(init_rccl PRIVATE /opt/rocm-7.0.2/lib /opt/rocm/lib)

# Link to MPI, HIP host runtime, and RCCL
target_link_libraries(init_rccl 
    MPI::MPI_CXX  # MPI libraries
    hip::host     # HIP host runtime
    ${RCCL_LIBRARY}  # RCCL library for collectives
)

# Set C++ standard for init_rccl
set_target_properties(init_rccl PROPERTIES 
    CXX_STANDARD 17 
    CXX_STANDARD_REQUIRED ON
)


# New executable for RCCL init test (local build, no install)
add_executable(test_clock_rccl test_clock_rccl.cpp)

# Reuse MPI and HIP (already found for test_boot)

# Find RCCL library explicitly
find_library(RCCL_LIBRARY 
    NAMES rccl librccl 
    PATHS /opt/rocm/lib /opt/rocm-7.0.2/lib  # Standard and versioned ROCm paths
    # NO_DEFAULT_PATH  # Allow system search as fallback
)

if(NOT RCCL_LIBRARY)
    message(FATAL_ERROR "RCCL library not found in /opt/rocm/lib or /opt/rocm-7.0.2/lib. Install rccl-dev or set ROCM_PATH.")
endif()

message(STATUS "Found RCCL library: ${RCCL_LIBRARY}")

# Include directories for MPI and HIP
target_include_directories(test_clock_rccl PRIVATE
    ${MPI_CXX_INCLUDE_DIRS}  # For MPI headers (C++ compatible)
    ${HIP_INCLUDE_DIRS}      # For HIP headers
    ${CMAKE_CURRENT_BINARY_DIR}/include     # For generated rocprofiler-sdk/defines.h
    ${PROJECT_SOURCE_DIR}/external/glog/src # For glog headers if needed
    ${MPI_C_INCLUDE_PATH}  # For MPI headers like mpi.h
    ${HIP_INCLUDE_DIRS}    # For HIP headers like hip_runtime.h
)

# Link directories for ROCm libs (to find librccl.so)
target_link_directories(test_clock_rccl PRIVATE /opt/rocm-7.0.2/lib /opt/rocm/lib)

# Link to MPI, HIP host runtime, and RCCL
target_link_libraries(test_clock_rccl 
    MPI::MPI_CXX  # MPI libraries
    hip::host     # HIP host runtime
    ${RCCL_LIBRARY}  # RCCL library for collectives
    rocprofiler-sdk::rocprofiler-sdk-common-library  # Correct target alias for common lib
    pthread
)

# Set C++ standard for init_rccl
set_target_properties(test_clock_rccl PROPERTIES 
    CXX_STANDARD 17 
    CXX_STANDARD_REQUIRED ON
)
