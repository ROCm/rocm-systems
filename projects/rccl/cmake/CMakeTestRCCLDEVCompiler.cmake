# CMakeTestRCCLDEVCompiler.cmake
# Verify the RCCLDEV compiler (rccl-device-compile) is functional.

if(NOT CMAKE_RCCLDEV_COMPILER_WORKS)
  execute_process(
    COMMAND "${CMAKE_RCCLDEV_COMPILER}" --version
    RESULT_VARIABLE _rccldev_result
    OUTPUT_VARIABLE _rccldev_output
    ERROR_QUIET
  )
  if(_rccldev_result EQUAL 0)
    set(CMAKE_RCCLDEV_COMPILER_WORKS TRUE CACHE INTERNAL "")
    message(STATUS "RCCLDEV compiler works: ${CMAKE_RCCLDEV_COMPILER}")
  else()
    message(FATAL_ERROR "RCCLDEV compiler test failed: ${CMAKE_RCCLDEV_COMPILER}")
  endif()
endif()
