#
#   Tests for the common library
#
project(rocprofiler-sdk-tests-common LANGUAGES C CXX)

set(common_sources
    c_array.cpp
    demangling.cpp
    environment.cpp
    md5sum.cpp
    mpl.cpp
    parse.cpp
    regex.cpp
    sha256.cpp
    uuid_v7.cpp)

add_executable(common-tests)
target_sources(common-tests PRIVATE ${common_sources})
target_link_libraries(
    common-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-output-library
            rocprofiler-sdk::rocprofiler-sdk-cereal
            GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET common-tests
    SOURCES ${common_sources}
    ENVIRONMENT
        "TEST_LOG_LEVEL=info"
        "LD_LIBRARY_PATH=${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/llvm/lib:/opt/rocm/lib:/opt/rocm/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )
