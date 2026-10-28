add_executable(core_tests
    test_chunk_framer.cpp
    test_error.cpp
    test_format_hex.cpp
    test_memory_sink.cpp
    test_module_map.cpp
    test_records.cpp
    test_reader_hardening.cpp
    test_self_exe.cpp
    test_sample_reconstructor.cpp)
target_link_libraries(core_tests PRIVATE cpu_snap::core cpu_snap_test_support GTest::gtest GTest::gtest_main)
target_compile_options(core_tests PRIVATE -Wall -Wextra -Wpedantic -O2 -g)
gtest_discover_tests(core_tests PROPERTIES TIMEOUT 10)
