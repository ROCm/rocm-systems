#
#
#

if(rocprofiler-sdk_BUILD_TREE OR rocprofiler-sdk-tests_BUILD_TREE)
    # unit tests are directly added in rocprofiler-sdk build tree.
    return()
endif()

# this package is from the installation
find_package(rocprofiler-sdk-tests)

# accommodate both hyphenated and underscored _FOUND names
if(rocprofiler_sdk_tests_FOUND OR rocprofiler-sdk-tests_FOUND)
    rocprofiler_sdk_tests_add_tests()
endif()
