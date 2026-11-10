obj-m := sdxi_char.o

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean

# Warn if io_uring_cmd support will be absent (CONFIG_IO_URING not set).
check-config:
	@grep -q '^CONFIG_IO_URING=y' $(KDIR)/.config 2>/dev/null || \
		echo "WARNING: CONFIG_IO_URING not set — io_uring_cmd path will not be built"
