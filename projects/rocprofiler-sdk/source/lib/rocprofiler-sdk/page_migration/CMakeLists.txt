#
#
set(ROCPROFILER_LIB_UVM_SOURCES page_migration.cpp)
set(ROCPROFILER_LIB_UVM_HEADERS defines.hpp page_migration.hpp utils.hpp)

target_sources(rocprofiler-sdk-object-library PRIVATE ${ROCPROFILER_LIB_UVM_SOURCES}
                                                      ${ROCPROFILER_LIB_UVM_HEADERS})
