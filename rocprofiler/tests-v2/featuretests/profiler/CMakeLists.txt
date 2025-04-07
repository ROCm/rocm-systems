# Handle HSA Interception Tool Tests
find_package(hsa-runtime64 REQUIRED CONFIG PATHS ${ROCM_PATH})

get_property(
    HSA_RUNTIME_INCLUDE_DIRECTORIES
    TARGET hsa-runtime64::hsa-runtime64
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(
    HSA_H hsa.h
    PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES hsa
    NO_DEFAULT_PATH REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
                    ${HSA_RUNTIME_INC_PATH} ${PROJECT_SOURCE_DIR}/tests-v2/featuretests)

find_package(
    Clang REQUIRED CONFIG
    PATHS "${ROCM_PATH}"
    PATH_SUFFIXES "llvm/lib/cmake/clang")

# Set the HIP language runtime link flags as FindHIP does not set them.
set(CMAKE_INSTALL_TESTDIR test/${PROJECT_NAME})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_PATH}/lib/cmake/hip")
set(CMAKE_HIP_ARCHITECTURES OFF)
if(NOT DEFINED HIP_ROOT_DIR)
    set(HIP_ROOT_DIR "${CMAKE_INSTALL_PREFIX}")
endif()
find_package(HIP REQUIRED MODULE)

# Enable CLANG-TIDY for static analysis search for clang-tidy
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(CLANG_TIDY_EXE)
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE}; -format-style='file';
                             -header-filter=${CMAKE_CURRENT_SOURCE_DIR};)
endif()

# ########################################################################################

function(rocprofiler_featuretests_profiler_add_test _TARGET)
    if(TARGET ${_TARGET})
        if(NOT TEST ${_TARGET})
            add_test(
                NAME ${_TARGET}
                COMMAND $<TARGET_FILE:${_TARGET}>
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
        endif()
        set_tests_properties(
            ${_TARGET} PROPERTIES LABELS "featuretests;profiler" ENVIRONMENT
                                  "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}" ${ARGN})
    endif()
endfunction()

function(rocprofiler_featuretests_profiler_add_executable _TARGET)
    hip_add_executable(${_TARGET} ${ARGN})
    rocprofiler_featuretests_profiler_add_test(${_TARGET})
endfunction()

# ########################################################################################
# App Based FeatureTests
# ########################################################################################

# installing the golden traces
file(
    GLOB files
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "apps/goldentraces/*.txt")
foreach(file ${files})
    configure_file(${file} ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/${file}
                   COPYONLY)
endforeach()

# Compile Applications hip_helloworld
set_source_files_properties(apps/hello_world_hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT
                                                                1)
rocprofiler_featuretests_profiler_add_executable(hip_helloworld apps/hello_world_hip.cpp)
set_target_properties(
    hip_helloworld PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                              "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(hip_helloworld PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS hip_helloworld
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)

# hip_vectoradd
set_source_files_properties(apps/vector_add_hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT
                                                               1)
rocprofiler_featuretests_profiler_add_executable(hip_vectoradd apps/vector_add_hip.cpp)
set_target_properties(
    hip_vectoradd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                             "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(hip_vectoradd PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS hip_vectoradd
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)

# mpi_vectoradd
add_library(rocprofiler-tests-mpi INTERFACE)
add_library(rocprofiler::tests-mpi ALIAS rocprofiler-tests-mpi)

find_package(MPI)
if(MPI_CXX_FOUND)
    set(USE_MPI 1)
    target_compile_definitions(rocprofiler-tests-mpi INTERFACE USE_MPI=1)
    target_link_libraries(rocprofiler-tests-mpi INTERFACE stdc++fs ${MPI_C_LIBRARIES})
    target_include_directories(rocprofiler-tests-mpi INTERFACE ${MPI_INCLUDE_PATH})
endif()

if(USE_MPI)
    set_source_files_properties(apps/vector_add_mpi.cpp
                                PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    rocprofiler_featuretests_profiler_add_executable(mpi_vectoradd
                                                     apps/vector_add_mpi.cpp)
    set_target_properties(
        mpi_vectoradd
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                   "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
    target_link_options(mpi_vectoradd PRIVATE "-Wl,--build-id=md5")
    install(
        TARGETS mpi_vectoradd
        RUNTIME
            DESTINATION
                ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
            COMPONENT tests)
    target_link_libraries(mpi_vectoradd PRIVATE rocprofiler::tests-mpi)
endif()

# openmp_helloworld
add_library(rocprofiler-tests-openmp INTERFACE)
add_library(rocprofiler::tests-openmp ALIAS rocprofiler-tests-openmp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(USE_OpenMP 1)
    target_compile_definitions(rocprofiler-tests-openmp INTERFACE USE_OpenMP=1)
    target_link_libraries(rocprofiler-tests-openmp INTERFACE OpenMP::OpenMP_CXX
                                                             hip::device)
elseif()
    message(WARNING "OpenMP not found")
    unset(USE_OpenMP)
endif()

# if(USE_OpenMP)
#     set_source_files_properties(apps/hello_world_omp.cpp
#                                 PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
#     rocprofiler_featuretests_profiler_add_executable(openmp_helloworld
#                                                      apps/hello_world_omp.cpp)
#     set_target_properties(
#         openmp_helloworld
#         PROPERTIES RUNTIME_OUTPUT_DIRECTORY
#                    "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")

#     target_link_options(openmp_helloworld PRIVATE "-Wl,--build-id=md5")
#     target_link_libraries(openmp_helloworld PRIVATE rocprofiler::tests-openmp)

#     install(
#         TARGETS openmp_helloworld
#         RUNTIME
#             DESTINATION
#                 ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
#             COMPONENT tests)
# endif()

# pmc correctness vectoradd
set_source_files_properties(apps/vectoradd.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_featuretests_profiler_add_executable(pmc_vectoradd apps/vectoradd.cpp)
set_target_properties(
    pmc_vectoradd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                             "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(pmc_vectoradd PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS pmc_vectoradd
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)

# pmc correctness hstogram
set_source_files_properties(apps/histogram.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_featuretests_profiler_add_executable(pmc_histogram apps/histogram.cpp)
set_target_properties(
    pmc_histogram PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                             "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(pmc_histogram PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS pmc_histogram
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)

# pmc correctness transpose
set_source_files_properties(apps/transpose.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_featuretests_profiler_add_executable(pmc_transpose apps/transpose.cpp)
set_target_properties(
    pmc_transpose PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                             "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(pmc_transpose PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS pmc_transpose
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)

# ATT correctness vectoradd
set_source_files_properties(apps/att_vectoradd.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_featuretests_profiler_add_executable(att_vectoradd apps/att_vectoradd.cpp)
set_target_properties(
    att_vectoradd PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                             "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(att_vectoradd PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS att_vectoradd
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)

# hsa-mem_async_copy -- Not Enabled for Now
set_source_files_properties(apps/async_mem_copy.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT
                                                               1)
rocprofiler_featuretests_profiler_add_executable(hsa_async_mem_copy
                                                 apps/async_mem_copy.cpp)
set_target_properties(
    hsa_async_mem_copy
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
               "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(hsa_async_mem_copy PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS hsa_async_mem_copy
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)
target_link_libraries(hsa_async_mem_copy hsa-runtime64::hsa-runtime64 Threads::Threads dl
                      stdc++fs)

# Setup testing
enable_testing()
find_package(GTest REQUIRED)

set(CORE_HSA_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa)
file(GLOB CORE_HSA_SRC_FILES ${CORE_HSA_DIR}/hsa_common.cpp)
set(CORE_HW_DIR ${PROJECT_SOURCE_DIR}/src/core/hardware)
file(GLOB CORE_HW_SRC_FILES ${CORE_HW_DIR}/hsa_info.cpp)
set(CORE_UTILS_DIR ${PROJECT_SOURCE_DIR}/src/utils)
file(GLOB CORE_UTILS_SRC_FILES ${CORE_UTILS_DIR}/helper.cpp)
set(TEST_UTILS_DIR ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/utils)
file(GLOB TEST_UTILS_SRC_FILES ${TEST_UTILS_DIR}/*.cpp)
set(GTEST_MAIN_DIR ${PROJECT_SOURCE_DIR}/tests-v2/featuretests)
file(GLOB GTEST_MAIN_SRC_FILE ${GTEST_MAIN_DIR}/gtests_main.cpp)

set_source_files_properties(apps/multithreaded_testapp.cpp
                            PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
rocprofiler_featuretests_profiler_add_executable(
    multithreaded_testapp apps/multithreaded_testapp.cpp ../utils/test_utils.cpp)
target_include_directories(
    multithreaded_testapp
    PRIVATE ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/profiler/apps)
target_link_libraries(multithreaded_testapp hsa-runtime64::hsa-runtime64 Threads::Threads
                      dl stdc++fs amd_comgr)
set_target_properties(
    multithreaded_testapp
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
               "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
target_link_options(multithreaded_testapp PRIVATE "-Wl,--build-id=md5")
install(
    TARGETS multithreaded_testapp
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)
add_dependencies(tests multithreaded_testapp)

# Multi-Queue Dependency Test
function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
    separate_arguments(
        CLANG_ARG_LIST
        UNIX_COMMAND
        "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
    add_custom_command(
        OUTPUT ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/${OUTPUT_FILE}
        COMMAND clang ${CLANG_ARG_LIST}
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/
        DEPENDS ${INPUT_FILE} clang
        COMMENT
            "Building ${OUTPUT_FILE}..." -E copy
            ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/${OUTPUT_FILE}
            ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/${target_id}_copy.hsaco
        VERBATIM)
    set(HSACO_TARGET_LIST
        ${HSACO_TARGET_LIST}
        ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/${OUTPUT_FILE}
        PARENT_SCOPE)
endfunction(generate_hsaco)

set(GPU_LIST "gfx900" "gfx906" "gfx908" "gfx90a" "gfx942" "gfx1030" "gfx1031" "gfx1032" "gfx1100" "gfx1101" "gfx1102" "gfx1150" "gfx1151" "gfx1200" "gfx1201")
foreach(target_id ${GPU_LIST})
    # generate kernel bitcodes
    generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/apps/copy.cl
                   ${target_id}_copy.hsaco)
    install(
        FILES
            "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/${target_id}_copy.hsaco"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests"
        COMPONENT tests)

endforeach(target_id)
add_custom_target(hsaco_targets DEPENDS ${HSACO_TARGET_LIST})

add_executable(multiqueue_testapp apps/multiqueue_testapp.cpp)
target_include_directories(
    multiqueue_testapp PRIVATE ${PROJECT_SOURCE_DIR}
                               ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/profiler)
# Link test executable against gtest & gtest_main
target_link_libraries(
    multiqueue_testapp PRIVATE rocprofiler-v2 hsa-runtime64::hsa-runtime64 GTest::gtest
                               GTest::gtest_main stdc++fs Threads::Threads amd_comgr dl)
add_dependencies(multiqueue_testapp hsaco_targets)
add_dependencies(tests multiqueue_testapp)
set_target_properties(
    multiqueue_testapp
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY
               "${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps")
install(
    TARGETS multiqueue_testapp
    RUNTIME
        DESTINATION
            ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps
        COMPONENT tests)

set(runFeatureTests_SOURCES
    profiler_gtest.cpp apps/hip_kernels.cpp ${GTEST_MAIN_SRC_FILE} ${CORE_HSA_SRC_FILES}
    ${CORE_HW_SRC_FILES} ${CORE_UTILS_SRC_FILES} ${TEST_UTILS_SRC_FILES})

set_source_files_properties(apps/hip_kernels.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(runFeatureTests ${runFeatureTests_SOURCES})

# link MPI and OpenMP to runFeatureTests for visibility
if(USE_MPI)
    target_compile_definitions(runFeatureTests PRIVATE USE_MPI=1)
endif()
if(USE_OpenMP)
    target_compile_definitions(runFeatureTests PRIVATE USE_OpenMP=1)
endif()

target_include_directories(
    runFeatureTests
    PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} ${PROJECT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/profiler)

# Link test executable against gtest & gtest_main
target_link_libraries(
    runFeatureTests PRIVATE rocprofiler-v2 hsa-runtime64::hsa-runtime64 GTest::gtest
                            GTest::gtest_main Threads::Threads dl stdc++fs amd_comgr)
add_dependencies(tests runFeatureTests)
target_link_options(runFeatureTests PRIVATE "-Wl,--build-id=md5")
install(TARGETS runFeatureTests
        RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests
                COMPONENT tests)

# add_test(AllTests runFeatureTests)
include(GoogleTest)

set(GTEST_DISCOVER_TESTS_TARGET runFeatureTests)
set(GTEST_DISCOVER_TESTS_LABELS "v2" "featuretests")
set(GTEST_DISCOVER_TESTS_ENVIRONMENT ${ROCPROFILER_MEMCHECK_PRELOAD_ENV})
configure_file(
    ${PROJECT_SOURCE_DIR}/cmake_modules/Templates/gtest_discover_tests_properties.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/runFeatureTests_TestProperties.cmake @ONLY)

# we cannot do LD_PRELOAD during test discovery but test discovery displays disabled tests
# in CDash so below is the work-around
# Commenting the first temporarily up till is fixed on Jenkins dockers
# if(NOT ROCPROFILER_MEMCHECK MATCHES "(Thread|Address)Sanitizer")
#     gtest_discover_tests(runFeatureTests)

#     set_property(
#         DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
#         APPEND
#         PROPERTY TEST_INCLUDE_FILES
#                  ${CMAKE_CURRENT_BINARY_DIR}/runFeatureTests_TestProperties.cmake)
# else()
    gtest_add_tests(
        TARGET runFeatureTests
        SOURCES "${runFeatureTests_SOURCES}"
        TEST_LIST runFeatureTests_TESTS)
    include(${CMAKE_CURRENT_BINARY_DIR}/runFeatureTests_TestProperties.cmake)
# endif()

# for the *_FilePlugin tests
if(NOT EXISTS "${PROJECT_BINARY_DIR}/test-output")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/test-output")
endif()

# Copy scripts, input files to samples folder
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/apps/goldentraces/basic_metrics.txt
               ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/apps/goldentraces/input.txt
               ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/apps/mpi_run.sh
               ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps/ COPYONLY)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/goldentraces/
    DESTINATION
        ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps/goldentraces
    COMPONENT tests)

if(NOT Python3_EXECUTABLE)
    find_package(
        Python3
        COMPONENTS Interpreter
        REQUIRED)
endif()

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pandas"
        RESULT_VARIABLE PANDAS_HEADER_PARSER
        OUTPUT_QUIET)
    if(NOT ${PANDAS_HEADER_PARSER} EQUAL 0)
        message(
                "The \"pandas\" Python3 package is not installed. \
                Please install it using the following command: \"${Python3_EXECUTABLE} -m pip install pandas\".")
    else()
        # cmake based tests
        include(${CMAKE_CURRENT_LIST_DIR}/counter_validation_tests.cmake)
        include(${CMAKE_CURRENT_LIST_DIR}/counter_correctness_tests.cmake)
        include(${CMAKE_CURRENT_LIST_DIR}/att_correctness_tests.cmake)
    endif()
