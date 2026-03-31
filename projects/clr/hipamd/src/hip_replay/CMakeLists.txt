cmake_minimum_required(VERSION 3.21)
project(hip_replay_tools LANGUAGES CXX)

# Find HIP - we only use the runtime API (no device code)
find_package(hip REQUIRED)

# Shared archive reader library
add_library(hrr_reader STATIC hrr_reader.cpp)
target_include_directories(hrr_reader PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(hrr_reader PUBLIC cxx_std_17)

# These tools are host-only (HIP runtime API, no device kernels).
# Link hip::host for headers + libamdhip64, but NOT hip::device
# which would add --offload-arch and try to compile as HIP.

# hrr-replay: Replay a .hrr trace archive
add_executable(hrr-replay hrr_replay.cpp)
target_link_libraries(hrr-replay PRIVATE hrr_reader hip::host)
target_include_directories(hrr-replay PRIVATE ${hip_INCLUDE_DIRS})
target_compile_features(hrr-replay PRIVATE cxx_std_17)

# hrr-bench: Benchmark and reproduce tool
add_executable(hrr-bench hrr_bench.cpp)
target_link_libraries(hrr-bench PRIVATE hrr_reader hip::host)
target_include_directories(hrr-bench PRIVATE ${hip_INCLUDE_DIRS})
target_compile_features(hrr-bench PRIVATE cxx_std_17)

# hrr-info: Archive info tool (no HIP dependency)
add_executable(hrr-info hrr_info.cpp)
target_link_libraries(hrr-info PRIVATE hrr_reader)
target_compile_features(hrr-info PRIVATE cxx_std_17)

# Install
install(TARGETS hrr-replay hrr-bench hrr-info
        RUNTIME DESTINATION bin)
