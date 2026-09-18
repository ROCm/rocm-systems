#
# tritonesque -- Mock compiler/kernel framework with protonesque profiler tool
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

add_subdirectory(protonesque)

rocprofiler_tests_configure_python_package_sources(tritonesque __init__.py hip_kernels.py)
