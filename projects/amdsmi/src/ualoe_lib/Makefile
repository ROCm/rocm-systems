LIB_NAME = ualoecfg
LIB_TARGET = lib$(LIB_NAME).a

SRCS = ualoe_lib.c ualoe_cdev.c ualoe_cb.c ualoe_nl.c

INCS = -I../include

OBJS = $(patsubst %.c, %.o, $(SRCS))

DEPS = libnl-3.0 libnl-genl-3.0
LIBS = $(shell pkg-config --libs $(DEPS))

CFLAGS = -Wall -Werror -c $(INCS) $(shell pkg-config --cflags $(DEPS))

# Unless specified otherwise, build with netlink support as first preference.
ifneq ("$(UALOE_NETLINK)", "n")
	CFLAGS += -DUALOE_NETLINK
endif

# Unless specified otherwise, enable logging by default.
ifneq ("$(UALOE_LOG)", "n")
	CFLAGS += -DUALOE_LOG_ENABLED_DEFAULT=1
else
	CFLAGS += -DUALOE_LOG_ENABLED_DEFAULT=0
endif

all: $(LIB_TARGET)

$(LIB_TARGET): $(OBJS)
	ar rcs $@ $(OBJS)

%.o: %.c
	gcc $(CFLAGS) $< -o $@ -lmnl $(LIBS)

clean:
	rm -f $(OBJS) $(LIB_TARGET)
