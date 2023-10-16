configure_file(basic_counters.xml
               ${PROJECT_BINARY_DIR}/share/rocprofiler/basic_counters.xml COPYONLY)
configure_file(derived_counters.xml
               ${PROJECT_BINARY_DIR}/share/rocprofiler/derived_counters.xml COPYONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/share/rocprofiler/basic_counters.xml
          ${PROJECT_BINARY_DIR}/share/rocprofiler/derived_counters.xml
    DESTINATION share/rocprofiler
    COMPONENT core)
