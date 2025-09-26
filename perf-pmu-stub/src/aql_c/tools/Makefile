# Makefile for packet generation tool

CC = gcc
CFLAGS = -Wall -Wextra -g -O2
INCLUDES = -I.. -I../..
LDFLAGS = -lm

# Source files
TOOL_SRC = packet_gen_tool.c
DEPS_SRC = ../packet_generation.c ../arch_creator.c ../gfx12_creator.c \
           ../pm4_packets.c ../counter_registry.c ../gfx12_events.c

# Object files
TOOL_OBJ = $(TOOL_SRC:.c=.o)
DEPS_OBJ = $(DEPS_SRC:.c=.o)

# Target
TARGET = packet_gen_tool

.PHONY: all clean install

all: $(TARGET)

$(TARGET): $(TOOL_OBJ) $(DEPS_OBJ)
	$(CC) $(TOOL_OBJ) $(DEPS_OBJ) -o $(TARGET) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(TARGET) $(TOOL_OBJ) $(DEPS_OBJ)

install: $(TARGET)
	install -m 755 $(TARGET) /usr/local/bin/

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build the packet generation tool"
	@echo "  clean   - Remove build artifacts"
	@echo "  install - Install tool to /usr/local/bin"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage example:"
	@echo "  make"
	@echo "  ./$(TARGET) gfx12 0x1000000 SQ:0:0x42 SQ:1:0x43"