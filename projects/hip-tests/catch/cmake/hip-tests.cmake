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
      add_executable(${_EXE_NAME} EXCLUDE_FROM_ALL ${SRC_NAME} ${COMMON_SHARED_SRC} Catch2::Catch2 $<TARGET_OBJECTS:Main_Object>)
      if(HIP_PLATFORM STREQUAL "amd")
        target_link_libraries(${_EXE_NAME} PUBLIC hiprtc)
      else()
        target_link_libraries(${_EXE_NAME} PUBLIC nvrtc)
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
      target_link_libraries(${_EXE_NAME} PUBLIC ${_LINKER_LIBS})
    endif()

    # Add dependency on build_tests to build it on this custom target
    add_dependencies(${_TEST_TARGET_NAME} ${_EXE_NAME})

    if (DEFINED _COMPILE_OPTIONS)
      target_compile_options(${_EXE_NAME} PUBLIC ${_COMPILE_OPTIONS})
    endif()
    target_link_libraries(${_EXE_NAME} PUBLIC Catch2::Catch2)

    foreach(arg IN LISTS _UNPARSED_ARGUMENTS)
        message(WARNING "Unparsed arguments: ${arg}")
    endforeach()
    # add binary to global list of binaries to install
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_EXE_TARGETS ${_EXE_NAME})
    catch_discover_tests("${_EXE_NAME}" DISCOVERY_MODE PRE_TEST PROPERTIES SKIP_REGULAR_EXPRESSION "HIP_SKIP_THIS_TEST")
    file(GLOB CTEST_INC_FILES "${CMAKE_CURRENT_BINARY_DIR}/${_EXE_NAME}-*_include.cmake")
    set_property(GLOBAL APPEND PROPERTY G_INSTALL_CTEST_INCLUDE_FILES ${CTEST_INC_FILES})
    if(NOT STANDALONE_TESTS EQUAL "1")
      break()
    endif()
  endforeach()
endfunction()

