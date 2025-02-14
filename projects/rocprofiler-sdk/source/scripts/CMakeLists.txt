configure_file(
    convert-counters-collection-format.py
    ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/convert-counters-collection-format.py
    COPYONLY)

set(CONVERT_CC_FORMAT_PATH
    ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/convert-counters-collection-format.py)

add_executable(rocprofiler-sdk::convert-counters-collection-format IMPORTED)
set_property(TARGET rocprofiler-sdk::convert-counters-collection-format
             PROPERTY IMPORTED_LOCATION ${CONVERT_CC_FORMAT_PATH})

install(
    FILES
        ${PROJECT_BINARY_DIR}/share/rocprofiler-sdk/convert-counters-collection-format.py
    DESTINATION share/rocprofiler-sdk
    COMPONENT core)
