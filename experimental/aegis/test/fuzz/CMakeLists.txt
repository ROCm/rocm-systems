# Fuzz tests (F-001, F-002)
#
# These tests use libFuzzer and require clang with -fsanitize=fuzzer support.
# Build with: cmake -DENABLE_FUZZ_TESTS=ON -DCMAKE_CXX_COMPILER=clang++ ..
#
# Run individual fuzzers:
#   ./fuzz_instruction_decoder corpus/instruction/ -max_total_time=60
#   ./fuzz_elf_parser corpus/elf/ -max_total_time=60
#
# The fuzzers will create a corpus of interesting inputs as they run.

#===----------------------------------------------------------------------===//
# Check if fuzzing is enabled and we're using Clang
#===----------------------------------------------------------------------===//

if(NOT ENABLE_FUZZ_TESTS)
  message(STATUS "Fuzz tests disabled (set -DENABLE_FUZZ_TESTS=ON to enable)")
  return()
endif()

# Check for Clang (required for libFuzzer)
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(WARNING "Fuzz tests require Clang compiler, skipping")
  return()
endif()

message(STATUS "Building fuzz tests with libFuzzer")

#===----------------------------------------------------------------------===//
# Common fuzzer settings
#===----------------------------------------------------------------------===//

set(FUZZ_COMPILE_FLAGS
  -fsanitize=fuzzer,address,undefined
  -fno-omit-frame-pointer
  -g
  -O1  # Use -O1 for better coverage feedback
)

set(FUZZ_LINK_FLAGS
  -fsanitize=fuzzer,address,undefined
)

#===----------------------------------------------------------------------===//
# F-001: Instruction Decoder Fuzzer
#===----------------------------------------------------------------------===//

add_executable(fuzz_instruction_decoder InstructionDecoderFuzz.cpp)
target_compile_options(fuzz_instruction_decoder PRIVATE ${FUZZ_COMPILE_FLAGS})
target_link_options(fuzz_instruction_decoder PRIVATE ${FUZZ_LINK_FLAGS})
target_link_libraries(fuzz_instruction_decoder
  aegisbit
)
target_include_directories(fuzz_instruction_decoder PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# F-002: ELF Parser Fuzzer
#===----------------------------------------------------------------------===//

add_executable(fuzz_elf_parser ElfParserFuzz.cpp)
target_compile_options(fuzz_elf_parser PRIVATE ${FUZZ_COMPILE_FLAGS})
target_link_options(fuzz_elf_parser PRIVATE ${FUZZ_LINK_FLAGS})
target_link_libraries(fuzz_elf_parser
  aegisbit
)
target_include_directories(fuzz_elf_parser PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${LLVM_INCLUDE_DIRS}
)

#===----------------------------------------------------------------------===//
# Copy corpus directories to build
#===----------------------------------------------------------------------===//

# Create corpus directories in build tree
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus/instruction)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus/elf)

# Copy seed files if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/corpus DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()
