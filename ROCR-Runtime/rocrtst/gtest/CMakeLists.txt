#
# Source files for Google Test Framework
#
set (gtFrwkSrcs src/gtest.cpp)
set (gtFrwkSrcs ${gtFrwkSrcs} src/gtest-port.cpp)
set (gtFrwkSrcs	${gtFrwkSrcs} src/gtest-printers.cpp)
set (gtFrwkSrcs	${gtFrwkSrcs} src/gtest-filepath.cpp)
set (gtFrwkSrcs	${gtFrwkSrcs} src/gtest-test-part.cpp)
set (gtFrwkSrcs	${gtFrwkSrcs} src/gtest-typed-test.cpp)
set (gtFrwkSrcs	${gtFrwkSrcs} src/gtest-death-test.cpp)
set (gtFrwkSrcs ${gtFrwkSrcs} src/gtest_main.cpp)

#
# Header files include path(s).
#
include_directories(include)
include_directories(include/gtest)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# Build Google Test Framework as a Static Library object
#
add_library(${GOOGLE_TEST_FRWK_NAME} STATIC ${gtFrwkSrcs})

#
# Install build artifacts into one common location
#
INSTALL(TARGETS ${GOOGLE_TEST_FRWK_NAME}
        ARCHIVE DESTINATION ${PROJECT_BINARY_DIR}/lib
        LIBRARY DESTINATION ${PROJECT_BINARY_DIR}/lib
        RUNTIME DESTINATION ${PROJECT_BINARY_DIR}/bin)
