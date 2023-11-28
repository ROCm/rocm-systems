#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-tests-kernel-tracing LANGUAGES CXX)

if(NOT TARGET rocprofiler::rocprofiler)
    find_package(rocprofiler REQUIRED)
endif()

add_library(kernel-tracing-test-tool SHARED)
target_sources(kernel-tracing-test-tool PRIVATE kernel-tracing.cpp)
target_link_libraries(
    kernel-tracing-test-tool
    PRIVATE rocprofiler::rocprofiler rocprofiler::cereal
            $<TARGET_NAME_IF_EXISTS:rocprofiler::tests-build-flags>)
set_target_properties(
    kernel-tracing-test-tool PROPERTIES INSTALL_RPATH "\$ORIGIN:\$ORIGIN/.."
                                        INSTALL_RPATH_USE_LINK_PATH ON)

install(
    TARGETS kernel-tracing-test-tool
    DESTINATION lib/rocprofiler
    COMPONENT rocprofiler-test-libs)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:kernel-tracing-test-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:kernel-tracing-test-tool>")
endif()

add_test(NAME test-kernel-tracing-execute COMMAND $<TARGET_FILE:reproducible-runtime>)

set_tests_properties(
    test-kernel-tracing-execute
    PROPERTIES
        TIMEOUT
        45
        LABELS
        "integration-tests"
        ENVIRONMENT
        "${PRELOAD_ENV};HSA_TOOLS_LIB=$<TARGET_FILE:rocprofiler::rocprofiler-shared-library>"
        FAIL_REGULAR_EXPRESSION
        "threw an exception")

foreach(FILENAME validate.py pytest.ini conftest.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

add_test(NAME test-kernel-tracing-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test-tool.json)

set_tests_properties(
    test-kernel-tracing-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS test-kernel-tracing-execute
               FAIL_REGULAR_EXPRESSION "threw an exception")
