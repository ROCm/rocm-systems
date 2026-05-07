# ROCm Kpack Runtime Tests

include(GoogleTest)

# Runtime tests - single executable with multiple translation units
add_executable(test_kpack_runtime
    test_kpack_api.cpp
    test_utils_test.cpp
    test_archive_integration.cpp
    test_loader_api.cpp
    test_isa_target_match.cpp
)
target_link_libraries(test_kpack_runtime
    PRIVATE
        rocm_kpack
        GTest::gtest
        GTest::gtest_main
)
# Include internal headers for white-box testing
target_include_directories(test_kpack_runtime
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# Apply ASAN to tests if enabled
if(ENABLE_ASAN)
    target_compile_options(test_kpack_runtime PRIVATE
        -fsanitize=address
        -fno-omit-frame-pointer
    )
    target_link_options(test_kpack_runtime PRIVATE
        -fsanitize=address
    )
endif()

# Discover tests and set environment variable for test data location
gtest_discover_tests(test_kpack_runtime
    PROPERTIES
        ENVIRONMENT "ROCM_KPACK_TEST_ASSETS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/test_assets"
)

# Compression test (will be implemented later)
# add_executable(test_compression test_compression.cpp)
# target_link_libraries(test_compression PRIVATE rocm_kpack GTest::gtest GTest::gtest_main)
# gtest_discover_tests(test_compression)
