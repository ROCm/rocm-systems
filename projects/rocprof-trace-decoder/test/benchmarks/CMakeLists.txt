# Copyright (c) 2026 Advanced Micro Devices, Inc.

add_executable(marker_benchmark marker_benchmark.cpp)
target_link_libraries(marker_benchmark PRIVATE rocprof-trace-decoder)
