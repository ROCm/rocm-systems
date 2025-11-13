# Minimum required c++ standard is 17 for compilation
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set cmake_prefix_path for searching, ROCM_PATH if avail otherwise default to general rocm install path
set(CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}" "/opt/rocm/")

# Define the directory to inspect
set(MY_DIR "$ENV{ROCM_PATH}/lib/cmake")

# Use file(GLOB ...) to get all files and subdirectories in MY_DIR
# GLOB_RECURSE can be used to include files in subdirectories recursively.
file(GLOB ALL_ITEMS "${MY_DIR}/*")

# Print the list of items found
message(STATUS "Contents of ${MY_DIR}:")
foreach(ITEM ${ALL_ITEMS})
    message(STATUS "- ${ITEM}")
endforeach()

# Define the directory to inspect
set(MY_DIR "$ENV{ROCM_PATH}/lib/cmake/rocprofiler-sdk")

# Use file(GLOB ...) to get all files and subdirectories in MY_DIR
# GLOB_RECURSE can be used to include files in subdirectories recursively.
file(GLOB ALL_ITEMS "${MY_DIR}/*")

# Print the list of items found
message(STATUS "Contents of ${MY_DIR}:")
foreach(ITEM ${ALL_ITEMS})
    message(STATUS "- ${ITEM}")
endforeach()

find_package(
    rocprofiler-sdk
    HINTS $ENV{ROCM_PATH}/lib/cmake /opt/rocm/lib/cmake
    CONFIG
    REQUIRED
)

add_library(rocprofiler-compute-tool SHARED)
target_sources(rocprofiler-compute-tool PRIVATE rocprofiler_compute_tool.cpp helper.cpp)
target_link_libraries(rocprofiler-compute-tool PRIVATE rocprofiler-sdk::rocprofiler-sdk)
