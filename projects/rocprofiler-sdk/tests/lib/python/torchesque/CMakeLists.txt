#
# torchesque -- Mock deep learning framework with kinetoesque profiler tool
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

add_subdirectory(kinetoesque)

rocprofiler_tests_configure_python_package_sources(torchesque __init__.py hip_kernels.py)
