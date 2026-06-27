cmake_minimum_required(VERSION 3.20)

project(aie_tests)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(hsa-runtime64 CONFIG REQUIRED NAMES hsa_runtime64 hsa-runtime64)

# error_callback_test: validates the per-queue error callback for AIE queues. Self-contained;
# it does not require any AIE kernel artifacts (no aiecc / NPU toolchain needed).
add_executable(error_callback_test error_callback_test.cc)
set_target_properties(error_callback_test
  PROPERTIES
    CXX_STANDARD                17
    CXX_EXTENSIONS              OFF
    CXX_VISIBILITY_PRESET       hidden
    VISIBILITY_INLINES_HIDDEN   ON
    POSITION_INDEPENDENT_CODE   ON
)
target_compile_options(error_callback_test
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
target_link_libraries(error_callback_test
  PRIVATE
    hsa-runtime64::hsa-runtime64
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(error_callback_test)
