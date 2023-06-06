# Set the HIP language runtime link flags as FindHIP does not set them.
set(CMAKE_INSTALL_TESTDIR test/${PROJECT_NAME})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_PATH}/lib/cmake/hip")
set(CMAKE_HIP_ARCHITECTURES OFF)
find_package(HIP REQUIRED MODULE)

# Setup testing
enable_testing()
find_package(GTest REQUIRED)

# installing the golden traces
file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "apps/goldentraces/*.txt")
foreach(file ${files})
    configure_file(${file} ${PROJECT_BINARY_DIR}/tests-v2/featuretests/tracer/${file} COPYONLY)
endforeach()

set(TEST_UTILS_DIR ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/utils)
file(GLOB TEST_UTILS_SRC_FILES ${TEST_UTILS_DIR}/*.cpp)
set(GTEST_MAIN_DIR ${PROJECT_SOURCE_DIR}/tests-v2/featuretests)
file(GLOB GTEST_MAIN_SRC_FILE ${GTEST_MAIN_DIR}/*.cpp)

# Compile Applications
# hip_helloworld
set_source_files_properties(apps/hello_world.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(tracer_hip_helloworld apps/hello_world.cpp)
set_target_properties(tracer_hip_helloworld PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests-v2/featuretests/tracer/apps")

#hsa-mem_async_copy and async_copy_on_engine
set_source_files_properties(apps/copy_on_engine.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(copy_on_engine apps/copy_on_engine.cpp)
set_target_properties(copy_on_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests-v2/featuretests/tracer/apps")
target_link_options(copy_on_engine PRIVATE "-Wl,--build-id=md5")
install(TARGETS copy_on_engine RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/tracer/apps COMPONENT tests)
target_link_libraries(copy_on_engine hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs)

# Add test cpp file
add_executable(runTracerFeatureTests tracer_gtest.cpp
 ${GTEST_MAIN_SRC_FILE}
 ${TEST_UTILS_SRC_FILES}
)

# Link test executable against gtest & gtest_main
target_include_directories(runTracerFeatureTests PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests-v2/featuretests/tracer)
target_link_libraries(runTracerFeatureTests PRIVATE hsa-runtime64::hsa-runtime64
                                              GTest::gtest GTest::gtest_main
                                              Threads::Threads dl stdc++fs)
add_dependencies(tests runTracerFeatureTests)
