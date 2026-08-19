# HRR Playback Tools
# Built as part of the main HIP runtime CMake tree.
# HIP headers are inherited from amdhip64 via target_link_libraries.

# hrr_api_args.h and capture headers are one directory up (hipamd/src/hrr/).
set(HRR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

# C++20 on Windows (MSVC requires it for some std features), C++17 on Linux.
if(WIN32)
  set(HRR_CXX_STANDARD 20)
else()
  set(HRR_CXX_STANDARD 17)
endif()

# ---------------------------------------------------------------------------
# hrr_reader — archive loader (STATIC)
# ---------------------------------------------------------------------------
add_library(hrr_reader STATIC hrr_reader.cpp)
set_target_properties(hrr_reader PROPERTIES CXX_STANDARD ${HRR_CXX_STANDARD} CXX_STANDARD_REQUIRED ON)
target_include_directories(hrr_reader PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HRR_DIR}
  ${HIP_COMMON_INCLUDE_DIR}
  ${PROJECT_BINARY_DIR}/include
)
target_compile_definitions(hrr_reader PRIVATE
  HRR_API_ARGS_IMPLEMENTATION
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)

# ---------------------------------------------------------------------------
# hrr-playback — full replay tool
# ---------------------------------------------------------------------------
add_executable(hrr-playback
  hrr_playback.cpp
  hip_playback.cpp
  hip_playback_generated.cpp
)
set_target_properties(hrr-playback PROPERTIES CXX_STANDARD ${HRR_CXX_STANDARD} CXX_STANDARD_REQUIRED ON)
target_include_directories(hrr-playback PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${HRR_DIR}
  ${HIP_COMMON_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include
)
target_compile_definitions(hrr-playback PRIVATE
  __HIP_PLATFORM_AMD__
  "HRR_BUILD_TYPE=\"$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unknown>\""
  $<$<BOOL:${WIN32}>:NOMINMAX WIN32_LEAN_AND_MEAN>
)
find_package(Threads REQUIRED)
target_link_libraries(hrr-playback PRIVATE hrr_reader amdhip64 Threads::Threads)

# Build identity reported by --version. Stamped by a build rule rather than at
# configure time so it cannot name a commit the binary was not built from; see
# HrrVersionStamp.cmake. Generated into the binary dir, which is already on the
# include path below.
set(HRR_BUILD_INFO_H ${CMAKE_CURRENT_BINARY_DIR}/hrr_build_info.h)
set(HRR_VERSION_STAMP ${CMAKE_CURRENT_SOURCE_DIR}/HrrVersionStamp.cmake)
set(HRR_VERSION_STAMP_ARGS
  -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
  -DOUT_FILE=${HRR_BUILD_INFO_H}
  -P ${HRR_VERSION_STAMP}
)
# Also generate once at configure time so the very first compile has the header
# regardless of how the generator orders the custom target.
execute_process(COMMAND ${CMAKE_COMMAND} ${HRR_VERSION_STAMP_ARGS})
add_custom_target(hrr_build_info ALL
  COMMAND ${CMAKE_COMMAND} ${HRR_VERSION_STAMP_ARGS}
  BYPRODUCTS ${HRR_BUILD_INFO_H}
  COMMENT "Stamping hrr-playback build revision"
)
add_dependencies(hrr-playback hrr_build_info)
target_include_directories(hrr-playback PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS hrr-playback RUNTIME DESTINATION bin)

# Crash-truncation recovery is covered by the hip-tests catch suite
# (projects/hip-tests/catch/unit/hrr/hrr_recovery.cc, run via HrrTest "[hrr]"),
# so it is part of CI rather than a standalone EXCLUDE_FROM_ALL target.
