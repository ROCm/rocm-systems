cmake_minimum_required(VERSION 3.15)

project(rocdxg_vram_budget_test LANGUAGES CXX)

enable_testing()

add_executable(rocdxg_vram_budget_test vram_budget_test.cpp)
target_compile_features(rocdxg_vram_budget_test PRIVATE cxx_std_11)
target_include_directories(
    rocdxg_vram_budget_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

add_test(NAME rocdxg_vram_budget_test COMMAND rocdxg_vram_budget_test)
