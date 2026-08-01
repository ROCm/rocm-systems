# Standalone experiments for dispatch-attached kernel timing. See README.md.
#
#   make ARCH=gfx950            build everything for a specific GPU
#   make ROCM_PATH=/opt/rocm    use a different toolchain

ROCM_PATH ?= /opt/rocm
ARCH      ?= gfx90a
HIPCC     := $(ROCM_PATH)/bin/hipcc

CXXFLAGS := -O2 -w --offload-arch=$(ARCH) -I$(ROCM_PATH)/include
LDFLAGS  := -L$(ROCM_PATH)/lib -lhsa-runtime64

PROGS := clock_domains ext_launch_gating marker_vs_attached rocprof_crosscheck \
         timestamp_discovery timestamp_validation launch_cost hsa_signal_crosscheck

all: $(PROGS)

# timestamp_validation and launch_cost sort candidate timings.
timestamp_validation launch_cost: CXXFLAGS += -include algorithm

%: %.cpp evtstamp.h
	$(HIPCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(PROGS)

.PHONY: all clean
