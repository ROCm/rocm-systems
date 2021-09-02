
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_executable(parallel-overhead parallel-overhead.cpp)
target_link_libraries(parallel-overhead Threads::Threads)
