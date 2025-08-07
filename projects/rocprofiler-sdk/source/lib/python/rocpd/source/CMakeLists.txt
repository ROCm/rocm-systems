#
# libpyrocpd python binding sources
#

set(libpyrocpd_source_headers common.hpp functions.hpp interop.hpp perfetto.hpp otf2.hpp
                              sql_generator.hpp pysqlite_Connection.h types.hpp)

set(libpyrocpd_source_sources functions.cpp interop.cpp otf2.cpp perfetto.cpp types.cpp)

foreach(_PYTHON_VERSION ${ROCPROFILER_PYTHON_VERSIONS})
    rocprofiler_rocpd_python_bindings_target_sources(
        ${_PYTHON_VERSION} PRIVATE ${libpyrocpd_source_sources}
        ${libpyrocpd_source_headers})
endforeach()

add_subdirectory(serialization)
