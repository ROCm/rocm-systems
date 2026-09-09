# HRR In-Tree Capture Layer
# Adds capture source files to the amdhip64 target.
# No separate library — capture is compiled into amdhip64 directly.

target_sources(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_metadata.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_writer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/hip_capture_generated.cpp
)

# Capture headers live here; the shared HRR archive wire format now lives in
# projects/hrr/include and is included as "hrr/hrr_api_args.h".
# Playback tools (hrr-playback, hrr_reader) now live in projects/hrr.
target_include_directories(amdhip64 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../../hrr/include
)
