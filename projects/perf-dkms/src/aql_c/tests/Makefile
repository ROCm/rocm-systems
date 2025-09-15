# Makefile for AQL C userspace tests
CC = gcc
CFLAGS = -Wall -Wextra -g -std=c99 -DDEBUG
INCLUDES = -I..

# Test executables
TESTS = test_pm4 test_counter_registry test_packet_generation

# PM4 test sources
TEST_PM4_SOURCES = test_pm4_packets.c
TEST_PM4_OBJECTS = test_pm4_packets.o
PM4_SOURCES = ../pm4_packets.c
PM4_OBJECTS = ../pm4_packets.o

# Counter registry test sources
TEST_COUNTER_SOURCES = test_counter_registry.c
TEST_COUNTER_OBJECTS = test_counter_registry.o
COUNTER_SOURCES = ../counter_registry.c ../gfx12_events.c ../arch_creator.c ../gfx12_creator.c
COUNTER_OBJECTS = ../counter_registry.o ../gfx12_events.o ../arch_creator.o ../gfx12_creator.o

# Packet generation test sources
TEST_PACKET_SOURCES = test_packet_generation.c
TEST_PACKET_OBJECTS = test_packet_generation.o
PACKET_SOURCES = ../packet_generation.c ../pm4_packets.c ../arch_creator.c ../gfx12_creator.c ../gfx12_events.c ../counter_registry.c
PACKET_OBJECTS = ../packet_generation.o ../pm4_packets.o ../arch_creator.o ../gfx12_creator.o ../gfx12_events.o ../counter_registry.o

# Build all tests
all: $(TESTS)

# PM4 test executable
test_pm4: $(TEST_PM4_OBJECTS) $(PM4_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Test executable created: test_pm4"

# Counter registry test executable
test_counter_registry: $(TEST_COUNTER_OBJECTS) $(COUNTER_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Test executable created: test_counter_registry"

# Packet generation test executable
test_packet_generation: $(TEST_PACKET_OBJECTS) $(PACKET_OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "Test executable created: test_packet_generation"

# Compile test sources
test_pm4_packets.o: test_pm4_packets.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

test_counter_registry.o: test_counter_registry.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

test_packet_generation.o: test_packet_generation.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Compile library sources (userspace mode)
../pm4_packets.o: ../pm4_packets.c
	$(CC) $(CFLAGS) $(INCLUDES) -DUSERSPACE_BUILD -c $< -o $@

../counter_registry.o: ../counter_registry.c
	$(CC) $(CFLAGS) $(INCLUDES) -DUSERSPACE_BUILD -c $< -o $@

../gfx12_events.o: ../gfx12_events.c
	$(CC) $(CFLAGS) $(INCLUDES) -DUSERSPACE_BUILD -c $< -o $@

../packet_generation.o: ../packet_generation.c
	$(CC) $(CFLAGS) $(INCLUDES) -DUSERSPACE_BUILD -c $< -o $@

../arch_creator.o: ../arch_creator.c
	$(CC) $(CFLAGS) $(INCLUDES) -DUSERSPACE_BUILD -c $< -o $@

../gfx12_creator.o: ../gfx12_creator.c
	$(CC) $(CFLAGS) $(INCLUDES) -DUSERSPACE_BUILD -c $< -o $@

# Run tests
run-tests: $(TESTS)
	@echo "Running PM4 packet tests..."
	./test_pm4
	@echo "Running counter registry tests..."
	./test_counter_registry
	@echo "Running packet generation tests..."
	./test_packet_generation

# Clean targets
clean:
	rm -f *.o ../*.o $(TESTS)

.PHONY: all run-tests clean