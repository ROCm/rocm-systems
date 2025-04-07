configure_file(counter_defs.yaml
               ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/counter_defs.yaml COPYONLY)
install(
    FILES ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/counter_defs.yaml
    DESTINATION share/rocprofiler-sdk
    COMPONENT core)
