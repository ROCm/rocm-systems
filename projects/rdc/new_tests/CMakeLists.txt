message("===============================================================")
message("              CMake RDC Diagnostics API Tests                  ")
message("===============================================================")

if(WIN32)
    message("RDC API tests are not supported on Windows platform")
    return()
endif()

# RPATH for finding libraries after installation
set(API_TEST_RPATH "\$ORIGIN" "\$ORIGIN/../lib" "\$ORIGIN/../lib/rdc")
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${API_TEST_RPATH})

message("")
message("Build Configuration:")
message("-----------BuildType: " ${BUILD_TYPE})
message("------------Compiler: " ${CMAKE_CXX_COMPILER})
message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
message("")

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RDC_API_TEST "rdc_api_test")

# Download and compile googletest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Test source files
set(testSources
    ${SRC_DIR}/main.cc
    ${SRC_DIR}/rdc_test_fixture.cc
    ${SRC_DIR}/field_metadata.cc
    ${SRC_DIR}/test_field_telemetry.cc
    ${SRC_DIR}/test_field_categories.cc
)

# Build rules
add_executable(${RDC_API_TEST} ${testSources})

# C++17 required
set_target_properties(${RDC_API_TEST} PROPERTIES CXX_STANDARD 17)

# Header file include paths
target_include_directories(
    ${RDC_API_TEST}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PUBLIC ${AMD_SMI_INCLUDE_DIR}
    PUBLIC ${SRC_DIR}
)

# Link libraries
target_link_libraries(
    ${RDC_API_TEST}
    PUBLIC rdc_bootstrap
    PUBLIC rdc
    PUBLIC GTest::gtest_main
    PUBLIC pthread
)

# Enable testing
enable_testing()
add_test(NAME RdcApiTest COMMAND ${RDC_API_TEST})

# Installation
install(
    TARGETS ${RDC_API_TEST}
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${RDC}/api_tests
    COMPONENT ${TESTS_COMPONENT}
)
