configure_file(basic_counters.xml
               ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/basic_counters.xml COPYONLY)
configure_file(derived_counters.xml
               ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/derived_counters.xml COPYONLY)

install(
    FILES ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/basic_counters.xml
          ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/derived_counters.xml
    DESTINATION share/rocprofiler-sdk
    COMPONENT core)
