# Common Tests - Test independent of all platforms
if(HIP_PLATFORM MATCHES "amd")
if(UNIX)
set(TEST_SRC
    gccTest.cc
    gpu.cpp
)
# Creating Custom object file
add_custom_command(OUTPUT LaunchKernel.o COMMAND gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/LaunchKernel.c -I${HIP_PATH}/include -D__HIP_PLATFORM_AMD__ -o LaunchKernel.o)
add_custom_target(LaunchKernel_custom DEPENDS LaunchKernel.o)
add_custom_command(OUTPUT hipMalloc.o COMMAND gcc -c ${CMAKE_CURRENT_SOURCE_DIR}/hipMalloc.c -I${HIP_PATH}/include -D__HIP_PLATFORM_AMD__ -o hipMalloc.o)
add_custom_target(hipMalloc_custom DEPENDS hipMalloc.o)

add_library(LaunchKernel_lib OBJECT IMPORTED)
add_library(hipMalloc_lib OBJECT IMPORTED)

set_property(TARGET LaunchKernel_lib PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/LaunchKernel.o")
set_property(TARGET hipMalloc_lib PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/hipMalloc.o")


hip_add_exe_to_target(NAME gccTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
		              LINKER_LIBS LaunchKernel_lib hipMalloc_lib)

add_dependencies(gccTests LaunchKernel_custom hipMalloc_custom)
endif()
endif()
