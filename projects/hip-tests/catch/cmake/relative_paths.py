import glob
import sys
import os
import re

if len(sys.argv) > 1:
    build_folder = sys.argv[1]
    install_folder = sys.argv[2]  
    #print(f"In relative_paths.py [1]: build folder: {build_folder}")
    #print(f"In relative_paths.py [2]: install folder: {install_folder}")
else:
    print(f"Args not provided. Please provide arg1-build folder-catch_tests, arg2-install folder")
    exit(1)

install_folder = os.path.abspath(install_folder)
install_catch_tests = os.path.join(install_folder, "catch_tests")
install_script = os.path.join(install_catch_tests, "script")

modified_msg = """ 
# File has been updated by hip-tests/catch folder for portability
"""
ctest_current_str = """
get_filename_component(FULL_FILE_PATH ${CMAKE_CURRENT_LIST_FILE} REALPATH)
get_filename_component(CTEST_CURRENT_DIR ${FULL_FILE_PATH} DIRECTORY)
get_filename_component(EXE_PATH ${CTEST_CURRENT_DIR}/.. REALPATH)
"""
if os.name == 'posix':
    library_path_str = """
set(LIB_PATH ${EXE_PATH}/../../../lib)
file(GLOB HIP_LIBS "${LIB_PATH}/libamdhip64*")
if(HIP_LIBS)
    set(LIB_PATH ${LIB_PATH} ${LIB_PATH}/rocm_sysdeps/lib)
else()
    set(LIB_PATH "/opt/rocm/lib")
endif()
"""
    ctest_current_str  = ctest_current_str + library_path_str

inc_cmake_pattern = "_include.cmake"
ctesttest_pattern = "CTestTestfile.cmake"

def make_test_files_portable(filenames):
    """
    changes the absolute paths with relative paths
    filenames - list of files to perform the workaround
    """
    for filename in filenames:
        try:
            filename = os.path.abspath(filename)
            # Read the entire content of the file
            with open(filename, 'r') as file:
                file_content = file.read()
            #print(f"**Done reading now parsing", filename)
            # 1 replace abs path with filename. Make relative
            old_text=os.path.dirname(os.path.abspath(filename))
            abs_dir_path = os.path.join(old_text, '')
            old_text = abs_dir_path.replace("\\", "/")
            if modified_msg not in file_content:
                file_content = modified_msg + file_content
            # Perform the replacement
            modified_content = file_content.replace(old_text, '')
            if inc_cmake_pattern in filename:
                # 2 get folder path relative to the current *_include.cmake  
                if r"get_filename_component(FULL_FILE_PATH" not in modified_content:
                    modified_content =  ctest_current_str + modified_content
                # 3 CTEST_FILE to have parameterized relative file
                test_cmake_pattern = r"\[==\[(\w+-[a-f0-9]+_tests.cmake)\]==\]"
                replace_test_pattern = r"${CTEST_CURRENT_DIR}/\1"
                modified_content = re.sub(test_cmake_pattern, replace_test_pattern, modified_content)
                # 4 script to use CatchAddTests.cmake from build rather than src
                add_test_pattern = r'include\(".*CatchAddTests\.cmake"\)'
                replace_add_test_pattern = 'include("${CTEST_CURRENT_DIR}/CatchAddTests.cmake")'
                modified_content = re.sub(add_test_pattern, replace_add_test_pattern, modified_content)
                # 5 use exe from previous folder
                exe_pattern = r"TEST_EXECUTABLE\s+\[==\[(.*?)\]==\]"
                replace_exe_pattern = r'TEST_EXECUTABLE  ${EXE_PATH}/\1'
                modified_content = re.sub(exe_pattern, replace_exe_pattern, modified_content)
                # 6 include _ctest.cmake file with path
                ctest_test_pattern = r'include\("(.*?_tests\.cmake)"\)'
                replace_ctest_pattern = r'include("${CTEST_CURRENT_DIR}/\1")'
                modified_content = re.sub(ctest_test_pattern, replace_ctest_pattern, modified_content)
                # 7 use script folder as cwd
                cwd_pattern = r"TEST_WORKING_DIR\s+\[==\[(.*?)\]==\]"
                replace_cwd_pattern = r'TEST_WORKING_DIR  ${EXE_PATH}'
                modified_content = re.sub(cwd_pattern, replace_cwd_pattern, modified_content)
                # 8 modify ld_library_path
                if os.name == 'posix':
                    lib_path_pattern = r"TEST_DL_PATHS\s+\[==\[(.*?)\]==\]"
                    replace_lib_path_pattern = r'TEST_DL_PATHS  ${LIB_PATH}'
                    modified_content = re.sub(lib_path_pattern, replace_lib_path_pattern, modified_content)

            filename = os.path.basename(filename)
            install_path = os.path.join(install_script, filename)
            # Write the modified content back to the file
            with open(install_path, 'w') as file:
                file.write(modified_content)
            #print(f"**Done parsing now writing into", install_path)
        except IOError as e:
            print(f"Error: '{e}'")
            sys.exit({e})
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit({e})

inccmake_files = glob.glob(build_folder + "/**/*"+inc_cmake_pattern, recursive=True)
make_test_files_portable(inccmake_files)
