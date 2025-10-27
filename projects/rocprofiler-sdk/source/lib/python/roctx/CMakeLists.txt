#
# CMakeLists.txt for roctx ctypes-based Python bindings
# This creates a pure Python package that works across all Python 3.x versions
#

# Set the installation directory - version agnostic! No more python3.8, python3.9, etc. -
# just one location
set(ROCTX_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python/site-packages/roctx"
    CACHE STRING "Installation directory for roctx ctypes Python package")

# Also set the build directory location for testing
set(ROCTX_BUILD_DIR "${CMAKE_BINARY_DIR}/${ROCTX_INSTALL_DIR}")

# Configure the __init__.py file to substitute CMake variables
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
               ${CMAKE_CURRENT_BINARY_DIR}/__init__.py.in @ONLY)

# Copy the configured file to the build directory
file(
    GENERATE
    OUTPUT ${ROCTX_BUILD_DIR}/__init__.py
    INPUT ${CMAKE_CURRENT_BINARY_DIR}/__init__.py.in)

# Copy other Python files to build directory for testing
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/context_decorators.py
     DESTINATION ${ROCTX_BUILD_DIR})

message(STATUS "")
message(STATUS "[roctx] Configuration:")
message(STATUS "[roctx]   Install directory: ${ROCTX_INSTALL_DIR}")
message(STATUS "[roctx]   Python version agnostic: YES")
message(STATUS "[roctx]   Binding type: ctypes (pure Python)")
message(STATUS "")

# Install the pure Python files
install(
    FILES ${ROCTX_BUILD_DIR}/__init__.py ${CMAKE_CURRENT_SOURCE_DIR}/context_decorators.py
    DESTINATION ${ROCTX_INSTALL_DIR}
    COMPONENT roctx)

# Create a symbolic link or install the shared library where Python can find it Option 1:
# Install to the same directory as the Python package
install(
    TARGETS rocprofiler-sdk-roctx-shared-library
    DESTINATION ${ROCTX_INSTALL_DIR}
    COMPONENT roctx
    OPTIONAL)

# Option 2: Create a lib subdirectory
install(
    TARGETS rocprofiler-sdk-roctx-shared-library
    DESTINATION ${ROCTX_INSTALL_DIR}/lib
    COMPONENT roctx
    OPTIONAL)

# Also install to standard location for system-wide use
install(
    TARGETS rocprofiler-sdk-roctx-shared-library
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT roctx
    OPTIONAL)

# Create a Python package marker
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__pycache__/.gitignore "*")

# Add test if we have Python for testing
if(ROCPROFILER_BUILD_TESTS)
    # Find any available Python3
    find_package(Python3 COMPONENTS Interpreter)

    if(Python3_FOUND)
        # Create a simple test script
        file(
            WRITE ${CMAKE_CURRENT_BINARY_DIR}/test_roctx.py
            "#!/usr/bin/env python3
import sys
import os

# Add the build site-packages directory to the path
sys.path.insert(0, '${CMAKE_BINARY_DIR}/lib/python/site-packages')

# Set LD_LIBRARY_PATH to find the shared library
os.environ['LD_LIBRARY_PATH'] = '${CMAKE_BINARY_DIR}/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import roctx
    print(f'SUCCESS: roctx module loaded')
    print(f'Version: {roctx.version_info}')
    print(f'Binding type: {roctx.version_info.get(\"binding_type\", \"unknown\")}')
    print(f'Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')

    # Test basic functionality
    roctx.mark('Test marker')
    print('SUCCESS: roctx.mark() works')

    level = roctx.rangePush('Test range')
    roctx.rangePop()
    print('SUCCESS: roctx.rangePush/Pop() works')

    tid = roctx.getThreadId()
    print(f'SUCCESS: roctx.getThreadId() returned {tid}')

    print('\\nAll basic tests passed!')
    sys.exit(0)
except Exception as e:
    print(f'FAIL: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
")

        # Add a test target
        add_test(
            NAME roctx_basic_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_roctx.py
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

        set_tests_properties(
            roctx_basic_test
            PROPERTIES
                ENVIRONMENT
                "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH};PYTHONPATH=${CMAKE_BINARY_DIR}/lib/python/site-packages"
                LABELS
                "python;roctx;ctypes")
    endif()
endif()
