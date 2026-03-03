message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
message("                  CMake AMD SMI Pytest                             ")
message("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

message("")
message("Build Configuration:")
# message("-----------BuildType: " ${CMAKE_BUILD_TYPE})
# message("------------Compiler: " ${CMAKE_CXX_COMPILER})
# message("-------------Version: " ${CMAKE_CXX_COMPILER_VERSION})
message("--------Proj Src Dir: " ${PROJECT_SOURCE_DIR})
# message("--------Proj Bld Dir: " ${PROJECT_BINARY_DIR})
# message("--------Proj Lib Dir: " ${PROJECT_BINARY_DIR}/lib)
# message("--------Proj Exe Dir: " ${PROJECT_BINARY_DIR}/bin)
message("--------Share Install Prefix: " ${SHARE_INSTALL_PREFIX})
message("--------Cpack_include_toplevel_directory: " ${CPACK_INCLUDE_TOPLEVEL_DIRECTORY})
message("--------CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY: " ${CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY})

# copy python test files into shared directory
install(
    DIRECTORY ./
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests/python_unittest/
    COMPONENT ${TESTS_COMPONENT}
    USE_SOURCE_PERMISSIONS FILES_MATCHING
    PATTERN "*.py")

# message(FATAL_ERROR "python lib stop")
