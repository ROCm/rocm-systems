#
# add sql common sources to output library target
#
set(output_sql_headers common.hpp deferred_transaction.hpp extract_data_type.hpp)
set(output_sql_sources common.cpp deferred_transaction.cpp)

target_sources(rocprofiler-sdk-output-library PRIVATE ${output_sql_sources}
                                                      ${output_sql_headers})
