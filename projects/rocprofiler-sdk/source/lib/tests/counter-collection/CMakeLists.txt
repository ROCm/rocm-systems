#
# aqlprofile counter collection tests using rocprofiler-sdk API
#
rocprofiler_deactivate_clang_tidy()

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    else()
        message(
            STATUS "aqlprofile counter collection tests: HIP compiler not found, skipping"
            )
        return()
    endif()
endif()

enable_language(HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

add_subdirectory(tool)

#
# test
#

include(GoogleTest)

set_source_files_properties(basic-tests.cpp PROPERTIES LANGUAGE HIP)

add_executable(aqlprofile-counter-collection-tests)
target_sources(aqlprofile-counter-collection-tests PRIVATE basic-tests.cpp)
target_include_directories(aqlprofile-counter-collection-tests
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    aqlprofile-counter-collection-tests
    PRIVATE rocprofiler-sdk-counters-tool rocprofiler-sdk::rocprofiler-sdk-shared-library
            rocprofiler-sdk::rocprofiler-sdk-fmt GTest::gtest GTest::gtest_main)

gtest_discover_tests(
    aqlprofile-counter-collection-tests DISCOVERY_MODE PRE_TEST
    TEST_PREFIX "aqlprofile.counter_collection."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES
    TIMEOUT 60 LABELS "integration-tests;aqlprofile")
