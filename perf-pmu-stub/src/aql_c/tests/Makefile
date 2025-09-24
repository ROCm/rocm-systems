# Userspace test Makefile for AQL architecture tests

# Compiler settings
CC := gcc
CFLAGS := -Wall -Wextra -g -std=c99 -I.. -I../..

# Source files for tests
TEST_SOURCES := test_gfx12.c test_dimension_decode.c
AQL_SOURCES := ../arch_creator.c ../gfx12_creator.c
ALL_SOURCES := $(TEST_SOURCES) $(AQL_SOURCES)

# Object files
TEST_OBJECTS := $(TEST_SOURCES:.c=.o)
AQL_OBJECTS := $(AQL_SOURCES:.c=.o)
ALL_OBJECTS := $(TEST_OBJECTS) $(AQL_OBJECTS)

# Test executables
TEST_EXECUTABLES := $(TEST_SOURCES:.c=)

# Default target
all: $(TEST_EXECUTABLES)
	@echo "All tests built successfully"

# Build test executables
test_gfx12: test_gfx12.o ../arch_creator.o ../gfx12_creator.o
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Built test_gfx12"

test_dimension_decode: test_dimension_decode.o
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Built test_dimension_decode"

# Object file rules
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# AQL source object files (build in parent directories)
../arch_creator.o: ../arch_creator.c
	$(CC) $(CFLAGS) -c -o $@ $<

../gfx12_creator.o: ../gfx12_creator.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Run tests
test: $(TEST_EXECUTABLES)
	@echo "Running test suite..."
	@for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
		echo ""; \
	done
	@echo "All tests completed successfully!"

# Run individual tests
run-gfx12: test_gfx12
	@echo "Running GFX12 tests..."
	./test_gfx12

run-dimension: test_dimension_decode
	@echo "Running dimension decode tests..."
	./test_dimension_decode

# Clean build artifacts
clean:
	rm -f $(ALL_OBJECTS) $(TEST_EXECUTABLES)
	rm -f ../*.o
	@echo "Cleaned test build artifacts"

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all test executables"
	@echo "  test         - Build and run all tests"
	@echo "  test_gfx12   - Build GFX12 test executable"
	@echo "  test_dimension_decode - Build dimension decode test executable"
	@echo "  run-gfx12    - Run GFX12 tests"
	@echo "  run-dimension - Run dimension decode tests"
	@echo "  clean        - Clean build artifacts"
	@echo "  help         - Show this help message"

# Phony targets
.PHONY: all test run-gfx12 run-dimension clean help

# Dependencies
test_gfx12.o: test_gfx12.c ../aql_structures.h ../arch_creator_common.h
test_dimension_decode.o: test_dimension_decode.c ../arch_creator_common.h
../arch_creator.o: ../arch_creator.c ../aql_structures.h ../arch_creator_common.h
../gfx12_creator.o: ../gfx12_creator.c ../aql_structures.h ../arch_creator_common.h