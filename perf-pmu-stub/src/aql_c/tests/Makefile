# Makefile for PM4 packet tests (userspace)
CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99 -DDEBUG
INCLUDES = -I..

# Test targets
TEST_SOURCES = test_pm4_packets.c
TEST_OBJECTS = test_pm4_packets.o
PM4_SOURCES = ../pm4_packets.c
PM4_OBJECTS = ../pm4_packets.o

# Main test executable
test_pm4: $(TEST_OBJECTS) $(PM4_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Test executable created: test_pm4"

# Compile test source
test_pm4_packets.o: test_pm4_packets.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Compile PM4 packets source (userspace mode)
../pm4_packets.o: ../pm4_packets.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Run tests
run-tests: test_pm4
	@echo "Running PM4 packet tests..."
	./test_pm4

# Clean targets
clean:
	rm -f *.o ../*.o test_pm4

.PHONY: run-tests clean