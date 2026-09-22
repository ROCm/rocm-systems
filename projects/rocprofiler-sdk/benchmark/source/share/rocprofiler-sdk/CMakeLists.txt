#
#
#

set(DATA_FILES benchmark_tables.sql benchmark_views.sql)

foreach(_FILE ${DATA_FILES})
    configure_file(${_FILE} ${CMAKE_DATAROOT_OUTPUT_DIRECTORY}/rocprofiler-sdk/${_FILE}
                   COPYONLY)
endforeach()

install(FILES ${DATA_FILES} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-sdk)
