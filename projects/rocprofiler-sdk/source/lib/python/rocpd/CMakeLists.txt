#
#
#

if(ROCPROFILER_BUILD_SQLITE3)
    message(
        FATAL_ERROR
            "rocpd python bindings require setting ROCPROFILER_BUILD_SQLITE3=OFF to interop with Python SQLite3 module"
        )
endif()

foreach(_PYTHON_VERSION ${ROCPROFILER_PYTHON_VERSIONS})
    rocprofiler_rocpd_python_bindings(${_PYTHON_VERSION})
endforeach()

add_subdirectory(source)

rocprofiler_reset_python3_cache()
