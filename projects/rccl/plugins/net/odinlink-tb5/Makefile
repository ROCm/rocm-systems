# SPDX-License-Identifier: MIT
#
# Copyright (c) OdinLink-Five contributors
# https://github.com/Geramy/OdinLink-Five
#
# OdinLink Thunderbolt 5 / USB4 RCCL net plugin
#
# Builds librccl-net-odl_tb5.so. Load with:
#   export NCCL_NET_PLUGIN=odl_tb5
#
PLUGIN_SO := librccl-net-odl_tb5.so
CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra
CPPFLAGS += -Iinclude -Inccl
LDFLAGS += -pthread -ldl

LIB_SRCS := \
	lib/odl_tb5_dev.c \
	lib/odl_tb5_xfer.c \
	lib/odl_tb5_peer.c \
	lib/odl_tb5_completion.c \
	lib/odl_tb5_stream.c

SRCS := plugin.c $(LIB_SRCS)

.PHONY: all build clean install

all: build

build: $(PLUGIN_SO)

$(PLUGIN_SO): $(SRCS) lib/odl_tb5_priv.h
	@printf "Compiling  %-35s > %s\n" plugin.c $@
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -shared -fvisibility=hidden \
		-o $@ $(SRCS) $(LDFLAGS)

install: $(PLUGIN_SO)
	@test -n "$(DESTDIR)$(PREFIX)" || { echo "Set PREFIX or DESTDIR"; exit 1; }
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 755 $(PLUGIN_SO) $(DESTDIR)$(PREFIX)/lib/$(PLUGIN_SO)

clean:
	rm -f $(PLUGIN_SO)
