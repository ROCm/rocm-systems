add_executable(vm_timeline_version_test vm_timeline_version_test.c)
target_include_directories(vm_timeline_version_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
add_test(NAME vm_timeline_version_test COMMAND vm_timeline_version_test)
set_tests_properties(vm_timeline_version_test PROPERTIES TIMEOUT 60)
