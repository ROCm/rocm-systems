# Set the HIP language runtime link flags as FindHIP does not set them.
set(CMAKE_INSTALL_TESTDIR test/${PROJECT_NAME})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_PATH}/lib/cmake/hip")
set(CMAKE_HIP_ARCHITECTURES OFF)
if(DEFINED ROCM_PATH)
    set(HIP_ROOT_DIR "${ROCM_PATH}/bin")
endif()
find_package(HIP REQUIRED MODULE)

# ########################################################################################
function(rocprofiler_featuretests_tracer_add_test _TARGET)
    if(TARGET ${_TARGET})
        if(NOT TEST ${_TARGET})
            add_test(
                NAME ${_TARGET}
                COMMAND $<TARGET_FILE:${_TARGET}>
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()

        set_tests_properties(
            ${_TARGET} PROPERTIES LABELS "featuretests;tracer" ENVIRONMENT
                                  "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}" ${ARGN})
    endif()
endfunction()

function(rocprofiler_featuretests_tracer_add_executable _TARGET)
    hip_add_executable(${_TARGET} ${ARGN})
    rocprofiler_featuretests_tracer_add_test(${_TARGET})
endfunction()

# Setup testing
enable_testing()
find_package(GTest REQUIRED)

# installing the golden traces
file(
    GLOB files
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "apps/goldentraces/*.txt")
foreach(file ${files})
    configure_file(${file} ${PROJECT_BINARY_DIR}/tests-v2/featuretests/tracer/${file}
                   COPYONLY)
endforeach()

set(TEST_UTILS_DIR ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/utils)
file(GLOB TEST_UTILS_SRC_FILES ${TEST_UTILS_DIR}/*.cpp)
set(GTEST_MAIN_DIR ${PROJECT_SOURCE_DIR}/tests-v2/featuretests)
file(GLOB GTEST_MAIN_SRC_FILE ${GTEST_MAIN_DIR}/*.cpp)

# Compile Applications hip_helloworld
set_source_files_properties(apps/hello_world.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_featuretests_tracer_add_executable(tracer_hip_helloworld apps/hello_world.cpp)
set_target_properties(
    tracer_hip_helloworld
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
               "${PROJECT_BINARY_DIR}/tests-v2/featuretests/tracer/apps")
target_link_options(tracer_hip_helloworld PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS tracer_hip_helloworld
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/tracer/apps
        COMPONENT tests)

# hsa-mem_async_copy and async_copy_on_engine
set_source_files_properties(apps/copy_on_engine.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT
                                                               1)
rocprofiler_featuretests_tracer_add_executable(copy_on_engine apps/copy_on_engine.cpp)
set_target_properties(
    copy_on_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              "${PROJECT_BINARY_DIR}/tests-v2/featuretests/tracer/apps")
target_link_options(copy_on_engine PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS copy_on_engine
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/tracer/apps
        COMPONENT tests)
target_link_libraries(copy_on_engine hsa-runtime64::hsa-runtime64 Threads::Threads dl
                      stdc++fs)

# Compile MatrixTranspose App with ROCTX
find_library(
    ROCTX_LIBRARY
    NAMES roctx64
    HINTS ${ROCM_PATH}/lib)
if(ROCTX_LIBRARY)
    set_source_files_properties(apps/MatrixTranspose.cpp
                                PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    hip_add_executable(tracer_matrix_transpose apps/MatrixTranspose.cpp)
    set_target_properties(
        tracer_matrix_transpose
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                   "${PROJECT_BINARY_DIR}/tests-v2/featuretests/tracer/apps")
    target_link_options(tracer_matrix_transpose PRIVATE "-Wl,--build-id=md5")
    target_include_directories(tracer_matrix_transpose PRIVATE ${ROCM_PATH})
    target_link_libraries(tracer_matrix_transpose ${ROCTX_LIBRARY})
    install(
        TARGETS tracer_matrix_transpose
        RUNTIME
            DESTINATION
                ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/tracer/apps
            COMPONENT tests)
endif()

# Add test cpp file
add_executable(runTracerFeatureTests tracer_gtest.cpp ${GTEST_MAIN_SRC_FILE}
                                     ${TEST_UTILS_SRC_FILES})

# Link test executable against gtest & gtest_main
target_include_directories(
    runTracerFeatureTests PRIVATE ${PROJECT_SOURCE_DIR}
                                  ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/tracer)
target_link_libraries(
    runTracerFeatureTests PRIVATE hsa-runtime64::hsa-runtime64 GTest::gtest
                                  GTest::gtest_main Threads::Threads dl stdc++fs)
target_link_options(runTracerFeatureTests PRIVATE "-Wl,--build-id=md5")
install(TARGETS runTracerFeatureTests
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests
                COMPONENT tests)
add_dependencies(tests runTracerFeatureTests)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/goldentraces/
    DESTINATION
        ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/tracer/apps/goldentraces
    COMPONENT tests)

if(NOT Python3_EXECUTABLE)
    find_package(
        Python3
        COMPONENTS Interpreter
        REQUIRED)
endif()

# cmake based tests
include(${CMAKE_CURRENT_LIST_DIR}/hiptrace_validation_tests.cmake)
