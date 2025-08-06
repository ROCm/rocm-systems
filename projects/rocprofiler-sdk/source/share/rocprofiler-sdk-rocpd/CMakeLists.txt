#
#
#

set(rocpd_schemas data_views.sql marker_views.sql rocpd_indexes.sql rocpd_tables.sql
                  rocpd_views.sql summary_views.sql)

foreach(_FILE ${rocpd_schemas})
    configure_file(${_FILE} ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk-rocpd/${_FILE}
                   COPYONLY)
    install(
        FILES ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk-rocpd/${_FILE}
        DESTINATION share/rocprofiler-sdk-rocpd
        COMPONENT rocpd)
endforeach()
