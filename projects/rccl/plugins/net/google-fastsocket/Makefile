RCCL_ROOT := ../../..
ROCM_PATH ?= /opt/rocm
# ?= would lose to make's built-in CXX default (g++); only replace that default.
ifeq ($(origin CXX),default)
CXX := $(ROCM_PATH)/bin/amdclang++
endif
INC := -I$(RCCL_ROOT)/plugins/net/example/nccl -I$(ROCM_PATH)/include -I$(CURDIR)
CXXFLAGS := -D__HIP_PLATFORM_AMD__ -include rccl_compat_v5.h
PLUGIN_SO := librccl-net.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): nccl-fastsocket/*.cc
	$(CXX) $(CXXFLAGS) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

nccl-fastsocket/*.cc:
	git clone https://github.com/google/nccl-fastsocket.git

install: $(BUILDDIR)/lib/$(PLUGIN_SO)

$(BUILDDIR)/lib/$(PLUGIN_SO): $(PLUGIN_SO)
	@printf "Grabbing %-35s > %s\n" $< $@
	mkdir -p $(BUILDDIR)/lib
	install -m 644 $< $@

clean:
	rm -f $(PLUGIN_SO)
	rm -Rf nccl-fastsocket

.PHONY: default install clean
