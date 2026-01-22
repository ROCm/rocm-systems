ROCM_PATH ?= $(wildcard /opt/rocm)
HIPCC = $(ROCM_PATH)/bin/hipcc

all: LatencyBench 

CXXFLAGS = -g -O0

LatencyBench: LatencyBench.cpp
	@printf "Compiling  %-35s > %s\n" $< $@
	$(HIPCC) $(CXXFLAGS) $^ -o $@

clean:
	rm -f *.o LatencyBench 
