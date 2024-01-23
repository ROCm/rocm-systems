foreach(FILENAME conftest.py pytest.ini)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}
                   ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()

add_subdirectory(input1)
add_subdirectory(input2)
