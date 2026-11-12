#
# Regression test: a HIP fat binary that is unregistered (its library dlclose'd) before any of its
# kernels are loaded must not be parsed by code object loads that happen afterwards.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

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
    endif()
endif()

project(rocprofiler-sdk-tests-hip-fatbin-unload LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(rocprofiler-sdk REQUIRED)

# Build the client tool library
add_library(hip-fatbin-unload-client SHARED)
target_sources(hip-fatbin-unload-client PRIVATE client.cpp)
target_link_libraries(hip-fatbin-unload-client PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                                       rocprofiler-sdk::tests-build-flags)

# HIP library that the test app loads and unloads without launching any of its kernels
set_source_files_properties(library.cpp PROPERTIES LANGUAGE HIP)
add_library(hip-fatbin-unload-library MODULE)
target_sources(hip-fatbin-unload-library PRIVATE library.cpp)
target_link_libraries(hip-fatbin-unload-library
                      PRIVATE rocprofiler-sdk::tests-build-flags)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
add_executable(hip-fatbin-unload-testapp)
target_sources(hip-fatbin-unload-testapp PRIVATE main.cpp)
target_link_libraries(hip-fatbin-unload-testapp
                      PRIVATE ${CMAKE_DL_LIBS} rocprofiler-sdk::tests-build-flags)
add_dependencies(hip-fatbin-unload-testapp hip-fatbin-unload-library)

# Register the test with CTest. The helper supplies the sanitizer preload, the
# integration-tests label and LD_LIBRARY_PATH. Deferred loading is pinned on because it is
# what keeps the library's fat binary from being parsed while the library is still loaded.
rocprofiler_add_integration_execute_test(
    test-hip-fatbin-unload
    COMMAND $<TARGET_FILE:hip-fatbin-unload-testapp>
            $<TARGET_FILE:hip-fatbin-unload-library>
    DEPENDS hip-fatbin-unload-client
    TIMEOUT 120
    PRELOAD "$<TARGET_FILE:hip-fatbin-unload-client>"
    ENVIRONMENT "HIP_ENABLE_DEFERRED_LOADING=1"
    PASS_REGULAR_EXPRESSION
        "Test PASSED: Successfully launched a kernel after unloading a HIP library!"
    FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}|SEGFAULT|Failed to \.*|Context is not valid|ERROR: \.*"
    SKIP_REGULAR_EXPRESSION "SKIP: \.*")
