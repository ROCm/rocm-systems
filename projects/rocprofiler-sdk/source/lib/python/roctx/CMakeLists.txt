#
#
#

foreach(_PYTHON_VERSION ${ROCPROFILER_PYTHON_VERSIONS})
    rocprofiler_roctx_python_bindings(${_PYTHON_VERSION})
endforeach()

rocprofiler_reset_python3_cache()
