#
# Unit tests for the rocprofv3 launcher (GPU-free)
#

find_package(rocprofiler_sdk_pytest REQUIRED)

if(NOT Python3_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
endif()

foreach(_FILE IN ITEMS conftest.py pytest.ini test_multipass_parsing.py
                       test_rocprof_attach.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${_FILE}
                   ${CMAKE_CURRENT_BINARY_DIR}/${_FILE} COPYONLY)
endforeach()

# Registered through rocprofiler_sdk_pytest_discover_tests rather than
# rocprofiler_add_integration_validate_test: these import the launcher and exercise its
# parsing helpers without a GPU or an application, so they take the unittests label
# instead of the integration-tests/validation labels that macro applies.
rocprofiler_sdk_pytest_discover_tests(
    rocprofv3-launcher-tests
    TEST_PATHS test_multipass_parsing.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
    EXTRA_ARGS --rocprofv3-path ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3
    PROPERTIES
    LABELS "unittests"
    TIMEOUT 45)

rocprofiler_sdk_pytest_discover_tests(
    rocprof-attach-launcher-tests
    TEST_PATHS test_rocprof_attach.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PYTHON_EXECUTABLE ${Python3_EXECUTABLE}
    EXTRA_ARGS --rocprof-attach-path
               ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprof-attach
    PROPERTIES
    LABELS "unittests"
    TIMEOUT 45)
