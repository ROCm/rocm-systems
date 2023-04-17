# Handle HSA Interception Tool Tests
find_package(hsa-runtime64 REQUIRED CONFIG PATHS ${ROCM_PATH})

get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HSA_H hsa.h
  PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
  PATH_SUFFIXES hsa
  NO_DEFAULT_PATH
  REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)

include_directories(${PROJECT_SOURCE_DIR} 
 ${CMAKE_CURRENT_SOURCE_DIR} 
 ${HSA_RUNTIME_INC_PATH} 
 ${PROJECT_SOURCE_DIR}/tests/featuretests)

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
   ${CLANG_TIDY_EXE};
   -format-style='file';
   -header-filter=${CMAKE_CURRENT_SOURCE_DIR};)
endif()
# ############################################################################################################################################
# App Based FeatureTests
# ############################################################################################################################################

# installing the golden traces
file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "apps/goldentraces/*.txt")
foreach(file ${files})
    configure_file(${file} ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/${file} COPYONLY)
endforeach()

# Compile Applications
# hip_helloworld
set_source_files_properties(apps/hello_world_hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(hip_helloworld apps/hello_world_hip.cpp)
set_target_properties(hip_helloworld PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps")
target_link_options(hip_helloworld PRIVATE "-Wl,--build-id=md5")
install(TARGETS hip_helloworld RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps COMPONENT tests)


#hip_vectoradd
set_source_files_properties(apps/vector_add_hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(hip_vectoradd apps/vector_add_hip.cpp)
set_target_properties(hip_vectoradd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps")
target_link_options(hip_vectoradd PRIVATE "-Wl,--build-id=md5")
install(TARGETS hip_vectoradd RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps COMPONENT tests)

#mpi_vectoradd
find_package(MPI)
if (MPI_CXX_FOUND)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set_source_files_properties(apps/vector_add_mpi.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(mpi_vectoradd apps/vector_add_mpi.cpp)
set_target_properties(mpi_vectoradd PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps")
target_link_options(mpi_vectoradd PRIVATE "-Wl,--build-id=md5")
install(TARGETS mpi_vectoradd RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps COMPONENT tests)
target_link_libraries(mpi_vectoradd ${MPI_C_LIBRARIES} stdc++fs)
endif()

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

#hsa-mem_async_copy -- Not Enabled for Now
set_source_files_properties(apps/async_mem_copy.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(hsa_async_mem_copy apps/async_mem_copy.cpp)
set_target_properties(hsa_async_mem_copy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps")
target_link_options(hsa_async_mem_copy PRIVATE "-Wl,--build-id=md5")
install(TARGETS hsa_async_mem_copy RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps COMPONENT tests)
target_link_libraries(hsa_async_mem_copy hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs)


# Setup testing
enable_testing()
find_package(GTest REQUIRED)

set(CORE_HSA_DIR ${PROJECT_SOURCE_DIR}/src/core/hsa)
file(GLOB CORE_HSA_SRC_FILES ${CORE_HSA_DIR}/hsa_common.cpp)
set(CORE_HW_DIR ${PROJECT_SOURCE_DIR}/src/core/hardware)
file(GLOB CORE_HW_SRC_FILES ${CORE_HW_DIR}/hsa_info.cpp)
set(CORE_UTILS_DIR ${PROJECT_SOURCE_DIR}/src/utils)
file(GLOB CORE_UTILS_SRC_FILES ${CORE_UTILS_DIR}/helper.cpp)
set(TEST_UTILS_DIR ${PROJECT_SOURCE_DIR}/tests/featuretests/utils)
file(GLOB TEST_UTILS_SRC_FILES ${TEST_UTILS_DIR}/*.cpp)
set(GTEST_MAIN_DIR ${PROJECT_SOURCE_DIR}/tests/featuretests)
file(GLOB GTEST_MAIN_SRC_FILE ${GTEST_MAIN_DIR}/gtests_main.cpp)

set_source_files_properties(apps/multithreaded_testapp.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(multithreaded_testapp apps/multithreaded_testapp.cpp ../utils/test_utils.cpp)
target_include_directories(multithreaded_testapp PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/featuretests/profiler/apps)
target_link_libraries(multithreaded_testapp hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs amd_comgr)
set_target_properties(multithreaded_testapp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps")
target_link_options(multithreaded_testapp PRIVATE "-Wl,--build-id=md5")
install(TARGETS multithreaded_testapp RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps COMPONENT tests)
add_dependencies(tests multithreaded_testapp)

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

set(GPU_LIST "gfx900" "gfx906" "gfx908" "gfx90a" "gfx1030")
foreach(target_id ${GPU_LIST})
  ## generate kernel bitcodes
  generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/apps/copy.cl ${target_id}_copy.hsaco)
# install(FILES "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/${target_id}_copy.hsaco"
#        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests"
#        COMPONENT tests)

endforeach(target_id)
add_custom_target(hsaco_targets DEPENDS ${HSACO_TARGET_LIST})

 add_executable(multiqueue_testapp  apps/multiqueue_testapp.cpp)
 target_include_directories(multiqueue_testapp PRIVATE ${PROJECT_SOURCE_DIR}  ${PROJECT_SOURCE_DIR}/inc ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
# Link test executable against gtest & gtest_main
 target_link_libraries(multiqueue_testapp PRIVATE ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64  GTest::gtest GTest::gtest_main stdc++fs Threads::Threads amd_comgr dl)
 add_dependencies(multiqueue_testapp hsaco_targets)
 add_dependencies(tests multiqueue_testapp )
 set_target_properties(multiqueue_testapp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps")
 install(TARGETS multiqueue_testapp RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps COMPONENT tests)

# add_executable(profiler_multiqueue_test discretetests/binary/multiqueue_test.cpp utils/csv_parser.cpp  utils/test_utils.cpp)
# target_include_directories(profiler_multiqueue_test PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)
# target_link_libraries(profiler_multiqueue_test PRIVATE hsa-runtime64::hsa-runtime64 Threads::Threads dl stdc++fs amd_comgr)
# add_dependencies(tests profiler_multiqueue_test)

# Add test cpp file
set_source_files_properties(profiler_gtest.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(runFeatureTests  profiler_gtest.cpp
 ${GTEST_MAIN_SRC_FILE}
 ${CORE_HSA_SRC_FILES}
 ${CORE_HW_SRC_FILES}
 ${CORE_UTILS_SRC_FILES}
 ${TEST_UTILS_SRC_FILES}
)

target_include_directories(runFeatureTests PRIVATE  ${TEST_DIR} 
 ${ROOT_DIR} 
 ${HSA_RUNTIME_INC_PATH} 
 ${PROJECT_SOURCE_DIR}  
 ${PROJECT_SOURCE_DIR}/tests/featuretests/profiler)

# Link test executable against gtest & gtest_main
target_link_libraries(runFeatureTests PRIVATE  ${ROCPROFILER_TARGET} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64
                                              GTest::gtest GTest::gtest_main 
                                              Threads::Threads dl stdc++fs amd_comgr)
add_dependencies(tests runFeatureTests)
target_link_options(runFeatureTests PRIVATE "-Wl,--build-id=md5")
install(TARGETS runFeatureTests RUNTIME DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests COMPONENT tests)
add_test(AllTests runFeatureTests)

# Copy scripts, input files to samples folder
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/apps/goldentraces/basic_metrics.txt ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/apps/goldentraces/input.txt ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/apps/mpi_run.sh ${PROJECT_BINARY_DIR}/tests/featuretests/profiler/apps/ COPYONLY)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/apps/goldentraces/
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests/featuretests/profiler/apps/goldentraces
  COMPONENT tests)
