# HRR In-Tree Capture Layer
# Adds capture source files to the amdhip64 target.
# No separate library — capture is compiled into amdhip64 directly.

target_sources(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_generated.cpp
)

# hrr_api_args.h and all capture headers live here.
target_include_directories(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Playback tools (hrr-playback, hrr_reader/hrr_playback libs)
add_subdirectory(playback)
