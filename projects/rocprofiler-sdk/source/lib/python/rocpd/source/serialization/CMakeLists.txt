#
# add sql common sources to output library target
#
set(libpyrocpd_source_serialization_sources sql.hpp perfetto.hpp)
set(libpyrocpd_source_serialization_headers sql.cpp)

rocprofiler_rocpd_python_bindings_object_sources(
    PRIVATE ${libpyrocpd_source_serialization_sources}
    ${libpyrocpd_source_serialization_headers})
