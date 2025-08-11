#
# Source files for Tests verifying rocrtst Utils library
#
set (rocrtstUtilsTestSrcs utils_timer_gtest.cpp)
set (rocrtstUtilsTestSrcs ${rocrtstUtilsTestSrcs} utils_timer_test.cpp)
set (rocrtstUtilsTestSrcs ${rocrtstUtilsTestSrcs} utils_cpp11_gtest.cpp)

#
# Header files include path(s).
#
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/utils)
include_directories(${PROJECT_SOURCE_DIR}/gtest/include)

#
# Build rule to build an executable object
#
add_executable(${ROCRTST_UTIL_TEST_NAME} ${rocrtstUtilsTestSrcs})

#
# Link unresolved symbols of rocrtst Utils Test executable
#
target_link_libraries(${ROCRTST_UTIL_TEST_NAME} ${ROCRTST_LIBS} elf c stdc++ dl pthread rt)

#
# Install build artifacts into one common location
#
INSTALL(TARGETS ${ROCRTST_UTIL_TEST_NAME}
        ARCHIVE DESTINATION ${PROJECT_BINARY_DIR}/lib
        LIBRARY DESTINATION ${PROJECT_BINARY_DIR}/lib
        RUNTIME DESTINATION ${PROJECT_BINARY_DIR}/bin)
