# Common test infrastructure
add_library(test_common INTERFACE)
target_include_directories(test_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_common INTERFACE hip::host)
