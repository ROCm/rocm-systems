cmake_minimum_required(VERSION 3.21)
project(test_roofline)

# Find Python (use -DPython3_EXECUTABLE to override if needed)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get Python library file (the actual .so file)
get_filename_component(PYTHON_LIB_FILE "${Python3_LIBRARY_RELEASE}" REALPATH)

# Build executable with RPATH=$ORIGIN
add_executable(test_roofline_cpp test_roofline.cpp)
target_include_directories(test_roofline_cpp PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(test_roofline_cpp PRIVATE ${Python3_LIBRARIES})
set_target_properties(test_roofline_cpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# Copy libpython.so next to executable after build
add_custom_command(TARGET test_roofline_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PYTHON_LIB_FILE}
        $<TARGET_FILE_DIR:test_roofline_cpp>
    COMMENT "Copying libpython to executable directory"
)

# Copy Python stdlib
add_custom_command(TARGET test_roofline_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${Python3_STDLIB}
        $<TARGET_FILE_DIR:test_roofline_cpp>/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}
    COMMENT "Copying Python standard library"
)

# Install Python dependencies from requirements.txt into bundled site-packages
add_custom_command(TARGET test_roofline_cpp POST_BUILD
    COMMAND ${Python3_EXECUTABLE} -m pip install
        --target $<TARGET_FILE_DIR:test_roofline_cpp>/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
        -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
    COMMENT "Installing Python dependencies to bundle"
)

# Copy test_roofline.py
add_custom_command(TARGET test_roofline_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/test_roofline.py
        $<TARGET_FILE_DIR:test_roofline_cpp>
    COMMENT "Copying test_roofline.py"
)
