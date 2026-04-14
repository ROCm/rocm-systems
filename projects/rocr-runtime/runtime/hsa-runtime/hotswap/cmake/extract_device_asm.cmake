# Extract the device assembly section from a hipcc -S output file.
# The device assembly is between the __CLANG_OFFLOAD_BUNDLE____START__ and
# __CLANG_OFFLOAD_BUNDLE____END__ markers for the amdgcn target.
file(READ "${INPUT}" content)

# Find the start marker
string(FIND "${content}" "# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn" start_pos)
if(start_pos EQUAL -1)
  message(FATAL_ERROR "Could not find device assembly bundle start in ${INPUT}")
endif()

# Skip the marker line itself
string(FIND "${content}" "\n" newline_pos REVERSE)
string(SUBSTRING "${content}" ${start_pos} -1 from_start)
string(FIND "${from_start}" "\n" first_newline)
math(EXPR body_start "${start_pos} + ${first_newline} + 1")

# Find the end marker
string(FIND "${content}" "# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn" end_pos)
if(end_pos EQUAL -1)
  message(FATAL_ERROR "Could not find device assembly bundle end in ${INPUT}")
endif()

math(EXPR length "${end_pos} - ${body_start}")
string(SUBSTRING "${content}" ${body_start} ${length} device_asm)

file(WRITE "${OUTPUT}" "${device_asm}")
message(STATUS "Extracted device assembly to ${OUTPUT}")
