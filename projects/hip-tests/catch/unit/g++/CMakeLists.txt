# AMD specific test
if(HIP_PLATFORM MATCHES "amd")
if(UNIX)
set(TEST_SRC
    hipMalloc.cc
)
# Creating Custom object file
add_custom_target(malloc_custom COMMAND g++ -c ${CMAKE_CURRENT_SOURCE_DIR}/hipMalloc.cpp -I${HIP_PATH}/include -D__HIP_PLATFORM_AMD__ -o malloc.o BYPRODUCTS malloc.o)
add_library(malloc_gpp OBJECT IMPORTED)
set_property(TARGET malloc_gpp PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/malloc.o")

hip_add_exe_to_target(NAME gppTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS malloc_gpp)

add_dependencies(gppTests malloc_custom)
endif()
endif()
