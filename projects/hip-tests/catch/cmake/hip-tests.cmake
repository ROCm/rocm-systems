include(Catch)

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
  hip_gen_exe_target(
    NAME ${_NAME}
    TEST_TARGET_NAME ${_TEST_TARGET_NAME}
    TEST_SRC ${_TEST_SRC}
    LINKER_LIBS ${_LINKER_LIBS}
    COMMON_SHARED_SRC ${_COMMON_SHARED_SRC}
    COMPILE_OPTIONS ${_COMPILE_OPTIONS}
    PROPERTY ${_PROPERTY}
    STANDALONE_FLAG 0
  )
  # If STANDALONE_TESTS==1, also generate per-file targets
  if(STANDALONE_TESTS EQUAL "1")
    hip_gen_exe_target(
      NAME ${_NAME}
      TEST_TARGET_NAME ${_TEST_TARGET_NAME}
      TEST_SRC ${_TEST_SRC}
      LINKER_LIBS ${_LINKER_LIBS}
      COMMON_SHARED_SRC ${_COMMON_SHARED_SRC}
      COMPILE_OPTIONS ${_COMPILE_OPTIONS}
      PROPERTY ${_PROPERTY}
      STANDALONE_FLAG 1
    )
  endif()

endfunction()

function(hip_gen_exe_target)
  set(options)
  set(args NAME TEST_TARGET_NAME PLATFORM COMPILE_OPTIONS STANDALONE_FLAG)
  set(list_args TEST_SRC LINKER_LIBS COMMON_SHARED_SRC PROPERTY)
  cmake_parse_arguments(
    PARSE_ARGV 0
    "" # variable prefix
    "${options}"
    "${args}"
    "${list_args}"
  )
  foreach(SRC_NAME ${TEST_SRC})

    if(NOT _STANDALONE_FLAG EQUAL "1")
      set(_EXE_NAME ${_NAME})
      set(SRC_NAME ${TEST_SRC})
    else()
      # strip extension of src and use exe name as src name
      get_filename_component(_EXE_NAME ${SRC_NAME} NAME_WLE)
      if(TARGET ${_EXE_NAME})
        message(WARNING "Duplicate per-file target name detected: ${_EXE_NAME}. Skipping this target!")
        continue()
      endif()
    endif()

    # Create shared lib of all tests
    set_source_files_properties(${SRC_NAME} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(${COMMON_SHARED_SRC} PROPERTIES LANGUAGE HIP)
    if(NOT RTC_TESTING)
      add_executable(${_EXE_NAME} EXCLUDE_FROM_ALL ${SRC_NAME} ${COMMON_SHARED_SRC} $<TARGET_OBJECTS:Main_Object> $<TARGET_OBJECTS:KERNELS>)
    else ()
      add_executable(${_EXE_NAME} EXCLUDE_FROM_ALL ${SRC_NAME} ${COMMON_SHARED_SRC} $<TARGET_OBJECTS:Main_Object>)
      if(HIP_PLATFORM STREQUAL "amd")
        target_link_libraries(${_EXE_NAME} hiprtc::hiprtc)
      else()
        target_link_libraries(${_EXE_NAME} nvrtc)
      endif()
    endif()
    set_target_properties(${_EXE_NAME} PROPERTIES LINKER_LANGUAGE HIP)

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
      #set_property(TARGET ${_EXE_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

    if(DEFINED _LINKER_LIBS)
      target_link_libraries(${_EXE_NAME} ${_LINKER_LIBS})
    endif()

    # Add dependency on build_tests to build it on this custom target
    add_dependencies(${_TEST_TARGET_NAME} ${_EXE_NAME})

    if (DEFINED _COMPILE_OPTIONS)
      target_compile_options(${_EXE_NAME} PUBLIC ${_COMPILE_OPTIONS})
    endif()
    target_link_libraries(${_EXE_NAME} Catch2::Catch2)
    target_link_libraries(${_EXE_NAME} hip::host hip::device)

    foreach(arg IN LISTS _UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed arguments: ${arg}")
    endforeach()
    # add binary to global list of binaries to install
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS ${_EXE_NAME})
    catch_discover_tests("${_EXE_NAME}" DISCOVERY_MODE PRE_TEST PROPERTIES SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
    file(GLOB CTEST_INC_FILES "${CMAKE_CURRENT_BINARY_DIR}/${_EXE_NAME}-*_include.cmake")
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CTEST_INCLUDE_FILES ${CTEST_INC_FILES})

    if(NOT _STANDALONE_FLAG EQUAL "1")
      break()
    endif()
  endforeach()
endfunction()