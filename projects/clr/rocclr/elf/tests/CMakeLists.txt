add_executable(clr-elfio-relocation-bounds-test relocation_bounds_test.cpp)
target_include_directories(clr-elfio-relocation-bounds-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(clr-elfio-relocation-bounds-test PRIVATE cxx_std_17)
add_test(NAME clr-elfio-relocation-bounds-test COMMAND clr-elfio-relocation-bounds-test)
