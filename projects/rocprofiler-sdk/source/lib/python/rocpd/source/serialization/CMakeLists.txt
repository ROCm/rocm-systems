#
# add sql common sources to output library target
#
set(libpyrocpd_source_serialization_sources sql.hpp)
set(libpyrocpd_source_serialization_headers sql.cpp)

foreach(_PYTHON_VERSION ${ROCPROFILER_PYTHON_VERSIONS})
    rocprofiler_rocpd_python_bindings_target_sources(
        ${_PYTHON_VERSION} PRIVATE ${libpyrocpd_source_serialization_sources}
        ${libpyrocpd_source_serialization_headers})
endforeach()
