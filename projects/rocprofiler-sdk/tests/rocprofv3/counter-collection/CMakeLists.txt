#
# Various counter collection tests
#

# copy to binary directory
rocprofiler_configure_pytest_files(COPY conftest.py CONFIG pytest.ini)

add_subdirectory(input1)
add_subdirectory(input2)
add_subdirectory(list_metrics)
