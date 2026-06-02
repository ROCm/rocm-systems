HIPCC ?= hipcc
HIPFLAGS ?= -g

SOURCES := $(wildcard *.hip)
TARGETS := $(SOURCES:.hip=)

all: $(TARGETS)

%: %.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
