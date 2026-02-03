#
# ROCProfiler SDK Python bindings for hardware counter collection
#

foreach(_PYTHON_VERSION ${ROCPROFILER_PYTHON_VERSIONS})
    rocprofiler_counter_python_bindings(${_PYTHON_VERSION})
    rocprofiler_python_preload_library(${_PYTHON_VERSION})
endforeach()

rocprofiler_reset_python3_cache()

# Add unit tests if testing is enabled
if(ROCPROFILER_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()
