# Makefile for packet generation and decoder tools

CC = gcc
CFLAGS = -Wall -Wextra -g -O2
INCLUDES = -I.. -I../..
LDFLAGS = -lm

# Packet generation tool
PACKET_GEN_SRC = packet_gen_tool.c
PACKET_GEN_DEPS = ../packet_generation.c ../arch_creator.c ../gfx12_creator.c \
                  ../pm4_packets.c ../counter_registry.c ../gfx12_events.c

# PM4 decoder tool (standalone)
PM4_DECODER_SRC = pm4_decoder.c

# Object files
PACKET_GEN_OBJ = $(PACKET_GEN_SRC:.c=.o)
PACKET_GEN_DEPS_OBJ = $(PACKET_GEN_DEPS:.c=.o)
PM4_DECODER_OBJ = $(PM4_DECODER_SRC:.c=.o)

# Targets
PACKET_GEN_TARGET = packet_gen_tool
PM4_DECODER_TARGET = pm4_decoder

.PHONY: all clean install

all: $(PACKET_GEN_TARGET) $(PM4_DECODER_TARGET)

$(PACKET_GEN_TARGET): $(PACKET_GEN_OBJ) $(PACKET_GEN_DEPS_OBJ)
	$(CC) $(PACKET_GEN_OBJ) $(PACKET_GEN_DEPS_OBJ) -o $(PACKET_GEN_TARGET) $(LDFLAGS)

$(PM4_DECODER_TARGET): $(PM4_DECODER_OBJ)
	$(CC) $(PM4_DECODER_OBJ) -o $(PM4_DECODER_TARGET)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(PACKET_GEN_TARGET) $(PM4_DECODER_TARGET) $(PACKET_GEN_OBJ) $(PACKET_GEN_DEPS_OBJ) $(PM4_DECODER_OBJ)

install: $(PACKET_GEN_TARGET) $(PM4_DECODER_TARGET)
	install -m 755 $(PACKET_GEN_TARGET) /usr/local/bin/
	install -m 755 $(PM4_DECODER_TARGET) /usr/local/bin/

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Build both packet generation and decoder tools"
	@echo "  clean   - Remove build artifacts"
	@echo "  install - Install tools to /usr/local/bin"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make"
	@echo "  ./$(PACKET_GEN_TARGET) gfx12 0x1000000 SQ:0:0x42 SQ:1:0x43"
	@echo "  echo 'c001790000000200e0000000' | ./$(PM4_DECODER_TARGET)"