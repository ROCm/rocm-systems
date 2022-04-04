set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

file(GLOB common_source ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)
add_library(openmp-common OBJECT ${common_source})
target_include_directories(openmp-common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

add_executable(openmp-cg ${CMAKE_CURRENT_SOURCE_DIR}/CG/cg.cpp
                         $<TARGET_OBJECTS:openmp-common>)
add_executable(openmp-lu ${CMAKE_CURRENT_SOURCE_DIR}/LU/lu.cpp
                         $<TARGET_OBJECTS:openmp-common>)

find_program(CLANGXX_EXECUTABLE NAMES clang++)
find_library(LIBOMP_LIBRARY
             NAMES omp ${CMAKE_SHARED_LIBRARY_PREFIX}omp${CMAKE_SHARED_LIBRARY_SUFFIX}.5)
if(CLANGXX_EXECUTABLE AND LIBOMP_LIBRARY)
    target_compile_options(openmp-common PUBLIC -W -Wall -fopenmp=libomp)
    target_compile_options(openmp-cg PRIVATE -W -Wall -fopenmp=libomp)
    target_link_libraries(openmp-cg PRIVATE ${LIBOMP_LIBRARY})
    target_compile_options(openmp-lu PRIVATE -W -Wall -fopenmp=libomp)
    target_link_libraries(openmp-lu PRIVATE ${LIBOMP_LIBRARY})
    omnitrace_custom_compilation(COMPILER ${CLANGXX_EXECUTABLE} TARGET openmp-common)
    omnitrace_custom_compilation(COMPILER ${CLANGXX_EXECUTABLE} TARGET openmp-cg)
    omnitrace_custom_compilation(COMPILER ${CLANGXX_EXECUTABLE} TARGET openmp-lu)
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(openmp-common PUBLIC OpenMP::OpenMP_CXX)
endif()
