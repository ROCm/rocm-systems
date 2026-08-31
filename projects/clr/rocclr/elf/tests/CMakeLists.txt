add_executable(clr-elfio-loader-bounds-test loader_bounds_test.cpp)
target_include_directories(clr-elfio-loader-bounds-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(clr-elfio-loader-bounds-test PRIVATE cxx_std_17)
add_test(NAME clr-elfio-loader-bounds-test COMMAND clr-elfio-loader-bounds-test)
