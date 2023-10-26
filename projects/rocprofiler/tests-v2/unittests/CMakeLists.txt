if(NOT ENABLE_ASAN_PACKAGING)
  add_subdirectory(profiler)
  add_subdirectory(core)
endif()
