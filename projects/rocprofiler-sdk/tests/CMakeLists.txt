#
#   Integration tests
#

# placeholder test
add_test(
    NAME cmake-version
    COMMAND ${CMAKE_COMMAND} --version
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
