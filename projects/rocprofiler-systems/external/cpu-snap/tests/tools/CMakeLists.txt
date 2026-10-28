# cpu-snap-run: pure launch-plan logic (header-only, no cpu_snap libraries needed).
add_executable(tools_tests test_launch_plan.cpp)
target_include_directories(tools_tests PRIVATE ${PROJECT_SOURCE_DIR}/tools/cpu_snap_run)
target_link_libraries(tools_tests PRIVATE GTest::gtest GTest::gtest_main)
target_compile_features(tools_tests PRIVATE cxx_std_20)
target_compile_options(tools_tests PRIVATE -Wall -Wextra -Wpedantic -O2 -g)
gtest_discover_tests(tools_tests PROPERTIES TIMEOUT 30)
