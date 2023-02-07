# Handle HSA Interception Tool Tests
find_package(hsa-runtime64 REQUIRED CONFIG PATHS ${ROCM_PATH})
file(GLOB PUBLIC_API_HEADERS "${PROJECT_SOURCE_DIR}/inc/*.h")
foreach(header ${PUBLIC_API_HEADERS})
  get_filename_component(header_subdir ${header} DIRECTORY)
  install(FILES ${header}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${header_subdir}
          COMPONENT runtime)
endforeach()

get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HSA_H hsa.h
  PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
  PATH_SUFFIXES hsa
  NO_DEFAULT_PATH
  REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)

include_directories(${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${HSA_RUNTIME_INC_PATH})

find_package(Clang REQUIRED CONFIG
             PATHS "${ROCM_PATH}"
             PATH_SUFFIXES "llvm/lib/cmake/clang")

# Set the HIP language runtime link flags as FindHIP does not set them.
set(CMAKE_INSTALL_TESTDIR test/${PROJECT_NAME})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${ROCM_PATH}/lib/cmake/hip")
set(CMAKE_HIP_ARCHITECTURES OFF)
find_package(HIP REQUIRED MODULE)

#Enable CLANG-TIDY for static analysis
# search for clang-tidy
find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if (CLANG_TIDY_EXE)
   set(CMAKE_CXX_CLANG_TIDY
   clang-tidy;
   -format-style='file';
   -header-filter=${CMAKE_CURRENT_SOURCE_DIR};
 )
endif()
# ############################################################################################################################################
# App Based FeatureTests
# ############################################################################################################################################

# installing the golden traces
file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "gtests/apps/goldentraces/*.txt")
foreach(file ${files})
    configure_file(${file} ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/${file} COPYONLY)
endforeach()

# Compile Applications
# hip_helloworld
set_source_files_properties(gtests/apps/hip/hello_world.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(hip_helloworld gtests/apps/hip/hello_world.cpp)
set_target_properties(hip_helloworld PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/gtests/apps")


#hip_vectoradd
set_source_files_properties(gtests/apps/hip/vector_add.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(hip_vectoradd gtests/apps/hip/vector_add.cpp)
set_target_properties(hip_vectoradd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/gtests/apps")


#openmp_helloworld
# find_package(hip REQUIRED)
# find_package(OpenMP)
# if(OpenMP_CXX_FOUND)
# # Source files.
# set_source_files_properties(gtests/apps/openmp/hello_world.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
# hip_add_executable(openmp_helloworld gtests/apps/openmp/hello_world.cpp)
# set_target_properties(openmp_helloworld PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/gtests/apps")

# # Link Libraries - HIP Device and OpenMP.
# target_compile_options(openmp_helloworld PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(openmp_helloworld PRIVATE hip::device ${OpenMP_CXX_FLAGS})
# endif()

#mpi_vectoradd
find_package(MPI)
if (MPI_CXX_FOUND)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set_source_files_properties(gtests/apps/mpi/vector_add.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(mpi_vectoradd gtests/apps/mpi/vector_add.cpp)
set_target_properties(mpi_vectoradd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/gtests/apps")
target_link_libraries(mpi_vectoradd ${MPI_C_LIBRARIES} stdc++fs)
endif()

#hsa-mem_async_copy -- Not Enabled for Now
# set_source_files_properties(gtests/apps/hsa/async_mem_copy.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
# hip_add_executable(hsa_async_mem_copy gtests/apps/hsa/async_mem_copy.cpp)
# set_target_properties(hsa_async_mem_copy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/gtests/apps")
# target_link_libraries(hsa_async_mem_copy hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs)


# Setup testing
enable_testing()
find_package(GTest REQUIRED)

# Add test cpp file
add_executable(runFeatureTests
 gtests/gtests_main.cpp
 utils/test_utils.cpp 
 gtests/apps/profiler_gtest.cpp
 gtests/apps/hip/hello_world_gtest.cpp
 gtests/apps/hip/vector_add_gtest.cpp
 gtests/apps/hsa/async_mem_copy_gtest.cpp     
 gtests/apps/mpi/vector_add_gtest.cpp
#  gtests/apps/openmp/hello_world_gtest.cpp
 gtests/functional/loadunload_gtest.cpp
)

target_include_directories(runFeatureTests PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
# Link test executable against gtest & gtest_main
target_link_libraries(runFeatureTests PRIVATE hsa-runtime64::hsa-runtime64
                                              GTest::gtest GTest::gtest_main 
                                              Threads::Threads dl stdc++fs)
add_dependencies(tests runFeatureTests)

# ############################################################################################################################################
# Functional Tests
# ############################################################################################################################################

file(GLOB ROCPROFILER_UTIL_SRC_FILES ${PROJECT_SOURCE_DIR}/src/utils/helper.cpp)

 set_source_files_properties(gtests/functional/multithread_gtest.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
# # Add test cpp file
 hip_add_executable(runFunctionalTests
  gtests/gtests_main.cpp
  ${ROCPROFILER_UTIL_SRC_FILES}
  utils/test_utils.cpp
  #gtests/functional/multithread_gtest.cpp
 )
 target_include_directories(runFunctionalTests PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
# # Link test executable against gtest & gtest_main
 target_link_libraries(runFunctionalTests PRIVATE ${ROCPROFILER_TARGET} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64
                                               GTest::gtest GTest::gtest_main 
                                               Threads::Threads dl stdc++fs amd_comgr)
 add_dependencies(tests runFunctionalTests)

# ############################################################################################################################################
# Discrete Tests
# ############################################################################################################################################

# Multi-Queue Dependency Test
function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/${OUTPUT_FILE}
    COMMAND clang ${CLANG_ARG_LIST}
    WORKING_DIRECTORY  ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/
    DEPENDS ${INPUT_FILE} clang
    COMMENT "Building ${OUTPUT_FILE}..."
    -E copy ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/${OUTPUT_FILE} ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/${target_id}_copy.hsaco
    VERBATIM)
  set(HSACO_TARGET_LIST ${HSACO_TARGET_LIST} ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/${OUTPUT_FILE} PARENT_SCOPE)
endfunction(generate_hsaco)

foreach(target_id ${GPU_TARGETS})
  ## generate kernel bitcodes
  generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/discretetests/binary/copy.cl ${target_id}_copy.hsaco)
endforeach(target_id)
add_custom_target(hsaco_targets DEPENDS ${HSACO_TARGET_LIST})

 add_executable(multiqueue_testapp  discretetests/binary/multiqueue_testapp.cpp)
 target_include_directories(multiqueue_testapp PRIVATE ${PROJECT_SOURCE_DIR}  ${PROJECT_SOURCE_DIR}/inc ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
# Link test executable against gtest & gtest_main
 target_link_libraries(multiqueue_testapp PRIVATE ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64  GTest::gtest GTest::gtest_main stdc++fs Threads::Threads amd_comgr dl)
 add_dependencies(multiqueue_testapp hsaco_targets)
 add_dependencies(tests multiqueue_testapp )

add_executable(profiler_multiqueue_test discretetests/binary/multiqueue_test.cpp utils/csv_parser.cpp  utils/test_utils.cpp)
target_include_directories(profiler_multiqueue_test PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
target_link_libraries(profiler_multiqueue_test PRIVATE hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs amd_comgr)
add_dependencies(tests profiler_multiqueue_test)

##### Profiler Discrete API Test #######
set_source_files_properties(discretetests/api/multithreaded_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable ( profiler_api_test discretetests/api/multithreaded_test.cpp utils/test_utils.cpp )
target_link_libraries ( profiler_api_test ${ROCPROFILER_TARGET} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs amd_comgr )
target_include_directories(profiler_api_test PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
add_dependencies(tests profiler_api_test)

set_source_files_properties(discretetests/api/att_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable ( profiler_api_att_test discretetests/api/att_test.cpp utils/test_utils.cpp )
target_link_libraries ( profiler_api_att_test ${ROCPROFILER_TARGET} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs )
target_include_directories(profiler_api_att_test PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
add_dependencies(tests profiler_api_att_test)

set_source_files_properties(discretetests/api/spm_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable ( profiler_api_spm_test discretetests/api/spm_test.cpp utils/test_utils.cpp )
target_link_libraries ( profiler_api_spm_test ${ROCPROFILER_TARGET} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs )
target_include_directories(profiler_api_spm_test PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
add_dependencies(tests profiler_api_spm_test)


##### Profiler Discrete Binary Tests #######
set_source_files_properties(discretetests/binary/multiprocess_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(profiler_multiprocess_test discretetests/binary/multiprocess_test.cpp utils/csv_parser.cpp  utils/test_utils.cpp)
target_include_directories(profiler_multiprocess_test PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
target_link_libraries(profiler_multiprocess_test PRIVATE hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs amd_comgr)
add_dependencies(tests profiler_multiprocess_test)

set_source_files_properties(discretetests/binary/multithreaded_testapp.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(multithreaded_testapp discretetests/binary/multithreaded_testapp.cpp utils/test_utils.cpp)
target_include_directories(multithreaded_testapp PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/featuretests/profiler)
target_link_libraries(multithreaded_testapp hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs amd_comgr)
add_dependencies(tests multithreaded_testapp)

set_source_files_properties(discretetests/binary/multithreaded_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(profiler_multithreaded_test discretetests/binary/multithreaded_test.cpp utils/csv_parser.cpp  utils/test_utils.cpp)
target_include_directories(profiler_multithreaded_test PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/featuretests/profiler)
target_link_libraries(profiler_multithreaded_test PRIVATE hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs amd_comgr)
add_dependencies(tests profiler_multithreaded_test)

add_test(AllTests runFeatureTests)

# Copy scripts, input files to samples folder
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/discretetests/run_discrete_tests.sh ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/ COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/discretetests/basic_metrics.txt ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/ COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/gtests/apps/mpi/mpi_run.sh ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/gtests/apps/ COPYONLY)
