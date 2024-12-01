# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Catch
-----

This module defines a function to help use the Catch test framework.

The :command:`catch_discover_tests` discovers tests by asking the compiled test
executable to enumerate its tests.  This does not require CMake to be re-run
when tests change.  However, it may not work in a cross-compiling environment,
and setting test properties is less convenient.

This command is intended to replace use of :command:`add_test` to register
tests, and will create a separate CTest test for each Catch test case.  Note
that this is in some cases less efficient, as common set-up and tear-down logic
cannot be shared by multiple test cases executing in the same instance.
However, it provides more fine-grained pass/fail information to CTest, which is
usually considered as more beneficial.  By default, the CTest test name is the
same as the Catch name; see also ``TEST_PREFIX`` and ``TEST_SUFFIX``.

.. command:: catch_discover_tests

  Automatically add tests with CTest by querying the compiled test executable
  for available tests::

    catch_discover_tests(target
                         [TEST_SPEC arg1...]
                         [EXTRA_ARGS arg1...]
                         [WORKING_DIRECTORY dir]
                         [TEST_PREFIX prefix]
                         [TEST_SUFFIX suffix]
                         [PROPERTIES name1 value1...]
                         [TEST_LIST var]
                         [REPORTER reporter]
                         [OUTPUT_DIR dir]
                         [OUTPUT_PREFIX prefix}
                         [OUTPUT_SUFFIX suffix]
    )

  ``catch_discover_tests`` sets up a post-build command on the test executable
  that generates the list of tests by parsing the output from running the test
  with the ``--list-test-names-only`` argument.  This ensures that the full
  list of tests is obtained.  Since test discovery occurs at build time, it is
  not necessary to re-run CMake when the list of tests changes.
  However, it requires that :prop_tgt:`CROSSCOMPILING_EMULATOR` is properly set
  in order to function in a cross-compiling environment.

  Additionally, setting properties on tests is somewhat less convenient, since
  the tests are not available at CMake time.  Additional test properties may be
  assigned to the set of tests as a whole using the ``PROPERTIES`` option.  If
  more fine-grained test control is needed, custom content may be provided
  through an external CTest script using the :prop_dir:`TEST_INCLUDE_FILES`
  directory property.  The set of discovered tests is made accessible to such a
  script via the ``<target>_TESTS`` variable.

  The options are:

  ``target``
    Specifies the Catch executable, which must be a known CMake executable
    target.  CMake will substitute the location of the built executable when
    running the test.

  ``TEST_SPEC arg1...``
    Specifies test cases, wildcarded test cases, tags and tag expressions to
    pass to the Catch executable with the ``--list-test-names-only`` argument.

  ``EXTRA_ARGS arg1...``
    Any extra arguments to pass on the command line to each test case.

  ``WORKING_DIRECTORY dir``
    Specifies the directory in which to run the discovered test cases.  If this
    option is not provided, the current binary directory is used.

  ``TEST_PREFIX prefix``
    Specifies a ``prefix`` to be prepended to the name of each discovered test
    case.  This can be useful when the same test executable is being used in
    multiple calls to ``catch_discover_tests()`` but with different
    ``TEST_SPEC`` or ``EXTRA_ARGS``.

  ``TEST_SUFFIX suffix``
    Similar to ``TEST_PREFIX`` except the ``suffix`` is appended to the name of
    every discovered test case.  Both ``TEST_PREFIX`` and ``TEST_SUFFIX`` may
    be specified.

  ``PROPERTIES name1 value1...``
    Specifies additional properties to be set on all tests discovered by this
    invocation of ``catch_discover_tests``.

  ``TEST_LIST var``
    Make the list of tests available in the variable ``var``, rather than the
    default ``<target>_TESTS``.  This can be useful when the same test
    executable is being used in multiple calls to ``catch_discover_tests()``.
    Note that this variable is only available in CTest.

  ``REPORTER reporter``
    Use the specified reporter when running the test case. The reporter will
    be passed to the Catch executable as ``--reporter reporter``.

  ``OUTPUT_DIR dir``
    If specified, the parameter is passed along as
    ``--out dir/<test_name>`` to Catch executable. The actual file name is the
    same as the test name. This should be used instead of
    ``EXTRA_ARGS --out foo`` to avoid race conditions writing the result output
    when using parallel test execution.

  ``OUTPUT_PREFIX prefix``
    May be used in conjunction with ``OUTPUT_DIR``.
    If specified, ``prefix`` is added to each output file name, like so
    ``--out dir/prefix<test_name>``.

  ``OUTPUT_SUFFIX suffix``
    May be used in conjunction with ``OUTPUT_DIR``.
    If specified, ``suffix`` is added to each output file name, like so
    ``--out dir/<test_name>suffix``. This can be used to add a file extension to
    the output e.g. ".xml".

#]=======================================================================]


#------------------------------------------------------------------------------
# TARGET_LIST TEST_SET
function(catch_discover_tests_compile_time_detection TARGET TEST_SET)
  cmake_parse_arguments(
    ""
    ""
    "TEST_PREFIX;TEST_SUFFIX;WORKING_DIRECTORY;TEST_LIST;REPORTER;OUTPUT_DIR;OUTPUT_PREFIX;OUTPUT_SUFFIX"
    "TEST_SPEC;EXTRA_ARGS;PROPERTIES"
    ${ARGN}
  )

  if(NOT _WORKING_DIRECTORY)
    set(_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  if(NOT _TEST_LIST)
    set(_TEST_LIST ${TARGET}_TESTS)
  endif()

  ## Generate a unique name based on the extra arguments
  string(SHA1 args_hash "${_TEST_SPEC} ${_EXTRA_ARGS} ${_REPORTER} ${_OUTPUT_DIR} ${_OUTPUT_PREFIX} ${_OUTPUT_SUFFIX}")
  string(SUBSTRING ${args_hash} 0 7 args_hash)

  # Define rule to generate test list for aforementioned test executable
  set(ctest_include_file "${CMAKE_CURRENT_BINARY_DIR}/${TEST_SET}_include-${args_hash}.cmake")
  set(ctest_tests_file "${CMAKE_CURRENT_BINARY_DIR}/${TEST_SET}_tests-${args_hash}.cmake")

  foreach(EXE_NAME ${TARGET})

    add_custom_command(
      TARGET ${EXE_NAME} POST_BUILD
      COMMAND "${CMAKE_COMMAND}"
              -D "TEST_TARGET=${EXE_NAME}"
              -D "TEST_EXECUTABLE=$<TARGET_FILE:${EXE_NAME}>"
              -D "TEST_EXECUTOR=${crosscompiling_emulator}"
              -D "TEST_WORKING_DIR=${_WORKING_DIRECTORY}"
              -D "TEST_SPEC=${_TEST_SPEC}"
              -D "TEST_EXTRA_ARGS=${_EXTRA_ARGS}"
              -D "TEST_PROPERTIES=${_PROPERTIES}"
              -D "TEST_PREFIX=${_TEST_PREFIX}"
              -D "TEST_SUFFIX=${_TEST_SUFFIX}"
              -D "TEST_LIST=${_TEST_LIST}"
              -D "TEST_REPORTER=${_REPORTER}"
              -D "TEST_OUTPUT_DIR=${_OUTPUT_DIR}"
              -D "TEST_OUTPUT_PREFIX=${_OUTPUT_PREFIX}"
              -D "TEST_OUTPUT_SUFFIX=${_OUTPUT_SUFFIX}"
              -D "CTEST_FILE=${ctest_tests_file}"
              -P "${_CATCH_DISCOVER_TESTS_SCRIPT}"
      VERBATIM
    )
  endforeach()

  file(RELATIVE_PATH ctestincludepath ${CMAKE_CURRENT_BINARY_DIR} ${ctest_include_file})
  file(RELATIVE_PATH ctestfilepath ${CMAKE_CURRENT_BINARY_DIR} ${ctest_tests_file})

  file(WRITE "${ctest_include_file}"
    "if(EXISTS \"${ctestfilepath}\")\n"
    "  include(\"${ctestfilepath}\")\n"
    "else()\n"
    "  message(WARNING \"Test ${TARGET} not built yet.\")\n"
    "endif()\n"
  )

  if(NOT ${CMAKE_VERSION} VERSION_LESS "3.10.0")
    # Add discovered tests to directory TEST_INCLUDE_FILES
    set_property(DIRECTORY
      APPEND PROPERTY TEST_INCLUDE_FILES "${ctestincludepath}"
    )
  else()
    # Add discovered tests as directory TEST_INCLUDE_FILE if possible
    get_property(test_include_file_set DIRECTORY PROPERTY TEST_INCLUDE_FILE SET)
    if (NOT ${test_include_file_set})
      set_property(DIRECTORY
        PROPERTY TEST_INCLUDE_FILE "${ctestincludepath}"
      )
    else()
      message(FATAL_ERROR
        "Cannot set more than one TEST_INCLUDE_FILE"
      )
    endif()
  endif()

endfunction()

###############################################################################




#------------------------------------------------------------------------------
# current staging
function(catch_discover_tests TARGET)
  cmake_parse_arguments(
    ""
    ""
    "TEST_PREFIX;TEST_SUFFIX;WORKING_DIRECTORY;TEST_LIST;REPORTER;OUTPUT_DIR;OUTPUT_PREFIX;OUTPUT_SUFFIX"
    "TEST_SPEC;EXTRA_ARGS;PROPERTIES"
    ${ARGN}
  )

  if(NOT _WORKING_DIRECTORY)
    set(_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  get_property(crosscompiling_emulator
    TARGET ${TARGET}
    PROPERTY CROSSCOMPILING_EMULATOR
  )
  ## Generate a unique name based on the extra arguments
  string(SHA1 args_hash "${_TEST_SPEC} ${_EXTRA_ARGS} ${_REPORTER} ${_OUTPUT_DIR} ${_OUTPUT_PREFIX} ${_OUTPUT_SUFFIX}")
  string(SUBSTRING ${args_hash} 0 7 args_hash)
  # Define rule to generate test list for aforementioned test executable
  set(ctest_include_file_build "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_include_build-${args_hash}.cmake")
  set(ctest_include_file_install "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_include_install-${args_hash}.cmake")
  set(ctest_tests_file_name "${TARGET}_tests-${args_hash}.cmake")
  set(ctest_tests_file "${CMAKE_CURRENT_BINARY_DIR}/${ctest_tests_file_name}")
  file(RELATIVE_PATH ctest_include_rel_path ${CMAKE_CURRENT_BINARY_DIR} ${ctest_include_file_build})
  file(RELATIVE_PATH ctest_file_rel_path ${CMAKE_CURRENT_BINARY_DIR} ${ctest_tests_file})
  file(RELATIVE_PATH _CATCH_ADD_TEST_SCRIPT ${CMAKE_CURRENT_BINARY_DIR} ${ADD_SCRIPT_PATH})
  file(RELATIVE_PATH CATCH_INCLUDE_PATH ${CMAKE_CURRENT_BINARY_DIR} ${CATCH_INCLUDE_PATH})
  if(NOT ${CMAKE_VERSION} VERSION_LESS "3.10.0")
      # write build time include file
      file(WRITE ${ctest_include_file_build} "set(_TARGET_EXECUTABLE ${TARGET})\n")
      file(APPEND ${ctest_include_file_build} "set(TARGET ${TARGET})\n")
      file(APPEND ${ctest_include_file_build} "set(_TEST_LIST ${TARGET}_TESTS)\n")
      file(APPEND ${ctest_include_file_build} "set(ctestfilepath ${ctest_file_rel_path})\n")
      file(APPEND ${ctest_include_file_build} "set(_CATCH_ADD_TEST_SCRIPT ${_CATCH_ADD_TEST_SCRIPT})\n")
      file(APPEND ${ctest_include_file_build} "set(crosscompiling_emulator ${crosscompiling_emulator})\n")
      file(APPEND ${ctest_include_file_build} "set(_PROPERTIES ${_PROPERTIES})\n")
      file(APPEND ${ctest_include_file_build} "include(${CATCH_INCLUDE_PATH})\n")
      # Add discovered tests to directory TEST_INCLUDE_FILES      
      set_property(DIRECTORY
        APPEND PROPERTY TEST_INCLUDE_FILES "${ctest_include_rel_path}"
      )
      
      # write install time include file
      file(WRITE ${ctest_include_file_install} "set(_TARGET_EXECUTABLE ${TARGET})\n")
      file(APPEND ${ctest_include_file_install} "set(TARGET ${TARGET})\n")
      file(APPEND ${ctest_include_file_install} "set(_TEST_LIST ${TARGET}_TESTS)\n")
      file(APPEND ${ctest_include_file_install} "set(ctestfilepath script/${ctest_tests_file_name})\n")
      file(APPEND ${ctest_include_file_install} "set(_CATCH_ADD_TEST_SCRIPT script/CatchAddTests.cmake)\n")
      file(APPEND ${ctest_include_file_install} "set(crosscompiling_emulator ${crosscompiling_emulator})\n")
      file(APPEND ${ctest_include_file_install} "set(_PROPERTIES ${_PROPERTIES})\n")
      file(APPEND ${ctest_include_file_install} "include(script/catch_include.cmake)\n")

      set_property(GLOBAL
        APPEND PROPERTY G_INSTALL_CTEST_INCLUDE_FILES "${ctest_include_file_install}"
      )
  endif()

endfunction()

###############################################################################

set(_CATCH_DISCOVER_TESTS_SCRIPT
  ${CMAKE_CURRENT_LIST_DIR}/CatchAddTests.cmake
  CACHE INTERNAL "Catch2 full path to CatchAddTests.cmake helper file"
)


###############################################################################
# function to be called by all tests
function(hip_add_exe_to_target_compile_time_detection)
  set(options)
  # NAME EventTest, TEST_SRC src, TEST_TARGET_NAME build_tests
  set(args NAME TEST_TARGET_NAME PLATFORM COMPILE_OPTIONS)
  set(list_args TEST_SRC LINKER_LIBS COMMON_SHARED_SRC PROPERTY)
  cmake_parse_arguments(
    PARSE_ARGV 0
    "" # variable prefix
    "${options}"
    "${args}"
    "${list_args}"
  )

  foreach(SRC_NAME ${TEST_SRC})
    if(NOT STANDALONE_TESTS EQUAL "1")
      set(_EXE_NAME ${_NAME})
      # take the entire source set for building the executable
      set(SRC_NAME ${TEST_SRC})
    else()
      # strip extension of src and use exe name as src name
      get_filename_component(_EXE_NAME ${SRC_NAME} NAME_WLE)
    endif()

    if(NOT RTC_TESTING)
      add_executable(${_EXE_NAME} EXCLUDE_FROM_ALL ${SRC_NAME} ${COMMON_SHARED_SRC} $<TARGET_OBJECTS:Main_Object> $<TARGET_OBJECTS:KERNELS>)
    else ()
      add_executable(${_EXE_NAME} EXCLUDE_FROM_ALL ${SRC_NAME} ${COMMON_SHARED_SRC} $<TARGET_OBJECTS:Main_Object>)
      if(HIP_PLATFORM STREQUAL "amd")
          target_link_libraries(${_EXE_NAME} hiprtc)
      else()
          target_link_libraries(${_EXE_NAME} nvrtc)
      endif()
    endif()



    if(UNIX)
      set(_LINKER_LIBS ${_LINKER_LIBS} stdc++fs)
      set(_LINKER_LIBS ${_LINKER_LIBS} -ldl)
    else()
      # res files are built resource files using rc files.
      # use llvm-rc exe to build the res files
      # Thes are used to populate the properties of the built executables
      if(EXISTS "${PROP_RC}/catchProp.res")
        set(_LINKER_LIBS ${_LINKER_LIBS} "${PROP_RC}/catchProp.res")
      endif()
      #set(_LINKER_LIBS ${_LINKER_LIBS} -noAutoResponse)
    endif()

    if(DEFINED _LINKER_LIBS)
      target_link_libraries(${_EXE_NAME} ${_LINKER_LIBS})
    endif()

    # Add dependency on build_tests to build it on this custom target
    add_dependencies(${_TEST_TARGET_NAME} ${_EXE_NAME})
    # add_dependencies(${_TEST_TARGET_NAME} ${_EXE_NAME})

    if (DEFINED _PROPERTY)
      set_property(TARGET ${_EXE_NAME} PROPERTY ${_PROPERTY})
    endif()

    if (DEFINED _COMPILE_OPTIONS)
      target_compile_options(${_EXE_NAME} PUBLIC ${_COMPILE_OPTIONS})
    endif()
    foreach(arg IN LISTS _UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed arguments: ${arg}")
    endforeach()
    get_property(crosscompiling_emulator
      TARGET ${_EXE_NAME}
      PROPERTY CROSSCOMPILING_EMULATOR
    )
    set(_EXE_NAME_LIST ${_EXE_NAME_LIST} ${_EXE_NAME})
    if(NOT STANDALONE_TESTS EQUAL "1")
      break()
    endif()
  endforeach()
  catch_discover_tests("${_EXE_NAME_LIST}" "${_NAME}" PROPERTIES  SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
endfunction()

###############################################################################
# current staging
# function to be called by all tests
function(hip_add_exe_to_target)
  set(options)
  set(args NAME TEST_TARGET_NAME PLATFORM COMPILE_OPTIONS)
  set(list_args TEST_SRC LINKER_LIBS COMMON_SHARED_SRC PROPERTY)
  cmake_parse_arguments(
    PARSE_ARGV 0
    "" # variable prefix
    "${options}"
    "${args}"
    "${list_args}"
  )
  foreach(SRC_NAME ${TEST_SRC})

    if(NOT STANDALONE_TESTS EQUAL "1")
      set(_EXE_NAME ${_NAME})
      set(SRC_NAME ${TEST_SRC})
    else()
      # strip extension of src and use exe name as src name
      get_filename_component(_EXE_NAME ${SRC_NAME} NAME_WLE)
    endif()

    # Create shared lib of all tests
    if(NOT RTC_TESTING)
      add_executable(${_EXE_NAME} EXCLUDE_FROM_ALL ${SRC_NAME} ${COMMON_SHARED_SRC} $<TARGET_OBJECTS:Main_Object> $<TARGET_OBJECTS:KERNELS>)
    else ()
      add_executable(${_EXE_NAME} EXCLUDE_FROM_ALL ${SRC_NAME} ${COMMON_SHARED_SRC} $<TARGET_OBJECTS:Main_Object>)
      if(HIP_PLATFORM STREQUAL "amd")
        target_link_libraries(${_EXE_NAME} hiprtc)
      else()
        target_link_libraries(${_EXE_NAME} nvrtc)
      endif()
    endif()
    if (DEFINED _PROPERTY)
      set_property(TARGET ${_EXE_NAME} PROPERTY ${_PROPERTY})
    endif()
    if(UNIX)
      set(_LINKER_LIBS ${_LINKER_LIBS} stdc++fs)
      set(_LINKER_LIBS ${_LINKER_LIBS} -ldl)
      set(_LINKER_LIBS ${_LINKER_LIBS} pthread)
      set(_LINKER_LIBS ${_LINKER_LIBS} rt)
    else()
      # res files are built resource files using rc files.
      # use llvm-rc exe to build the res files
      # Thes are used to populate the properties of the built executables
      if(EXISTS "${PROP_RC}/catchProp.res")
        set(_LINKER_LIBS ${_LINKER_LIBS} "${PROP_RC}/catchProp.res")
      endif()
    endif()

    if(DEFINED _LINKER_LIBS)
      target_link_libraries(${_EXE_NAME} ${_LINKER_LIBS})
    endif()

    # Add dependency on build_tests to build it on this custom target
    add_dependencies(${_TEST_TARGET_NAME} ${_EXE_NAME})

    if (DEFINED _COMPILE_OPTIONS)
      target_compile_options(${_EXE_NAME} PUBLIC ${_COMPILE_OPTIONS})
    endif()

    foreach(arg IN LISTS _UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed arguments: ${arg}")
    endforeach()
    # add binary to global list of binaries to install
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS ${_EXE_NAME})
    catch_discover_tests("${_EXE_NAME}" PROPERTIES SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")

    if(NOT STANDALONE_TESTS EQUAL "1")
      break()
    endif()

  endforeach()

endfunction()

