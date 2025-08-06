add_definitions(-DAMD_INTERNAL_BUILD=1)
add_custom_target(tests)
add_dependencies(tests rocprofiler_tool)
add_custom_target(
    check
    COMMAND ${PROJECT_BINARY_DIR}/run_tests.sh
    DEPENDS tests)
add_subdirectory(featuretests)
add_subdirectory(memorytests)
add_subdirectory(microbenchmarks)
add_subdirectory(HSAToolLibrary)
add_subdirectory(unittests)
configure_file(run_tests.sh ${PROJECT_BINARY_DIR} COPYONLY)
