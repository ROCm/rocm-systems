CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -g
TARGET   = sdxi-cli
SRCS     = sdxi-cli.c

$(TARGET): $(SRCS) sdxi_user.h
	$(CC) $(CFLAGS) -o $@ $(SRCS)

install: $(TARGET)
	install -m 0755 $(TARGET) /usr/local/bin/

clean:
	rm -f $(TARGET)

.PHONY: install clean
