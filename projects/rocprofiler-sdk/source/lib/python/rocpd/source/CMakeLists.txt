#
# libpyrocpd python binding sources
#

set(libpyrocpd_source_object_headers common.hpp functions.hpp perfetto.hpp otf2.hpp
                                     sql_generator.hpp types.hpp)

set(libpyrocpd_source_object_sources functions.cpp otf2.cpp perfetto.cpp types.cpp)

rocprofiler_rocpd_python_bindings_object_sources(
    PRIVATE ${libpyrocpd_source_object_headers} ${libpyrocpd_source_object_sources})

set(libpyrocpd_source_target_headers interop.hpp pysqlite_Connection.h)

set(libpyrocpd_source_target_sources interop.cpp)

foreach(_PYTHON_VERSION ${ROCPROFILER_PYTHON_VERSIONS})
    rocprofiler_rocpd_python_bindings_target_sources(
        ${_PYTHON_VERSION} PRIVATE ${libpyrocpd_source_target_sources}
        ${libpyrocpd_source_target_headers})
endforeach()

add_subdirectory(serialization)
