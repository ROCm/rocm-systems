# Simple Makefile for ROCm base applications
# Compiler and flags
CC = hipcc
CFLAGS = -std=c++11 -O2 -g
LDFLAGS = -lhsa-runtime64

# Application targets
APPS = app1_signal_create app2_signal_store app3_barrier_packet app4_dual_barrier app5_cpu_memory_pool app6_aql_start_packet app7_aql_read_packet app8_aql_stop_packet

# Default target
all: $(APPS)

# Individual application targets
app1_signal_create: app1_signal_create.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

app2_signal_store: app2_signal_store.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

app3_barrier_packet: app3_barrier_packet.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

app4_dual_barrier: app4_dual_barrier.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

app5_cpu_memory_pool: app5_cpu_memory_pool.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# AQL profiler applications (require aql_profiler.cpp and aqlprofile library)
app6_aql_start_packet: app6_aql_start_packet.cpp aql_profiler.cpp aql_profiler.h
	$(CC) $(CFLAGS) -I../../projects/aqlprofile/src/core/include -o $@ app6_aql_start_packet.cpp aql_profiler.cpp $(LDFLAGS) -L/opt/rocm/lib -lhsa-amd-aqlprofile64

app7_aql_read_packet: app7_aql_read_packet.cpp aql_profiler.cpp aql_profiler.h
	$(CC) $(CFLAGS) -I../../projects/aqlprofile/src/core/include -o $@ app7_aql_read_packet.cpp aql_profiler.cpp $(LDFLAGS) -L/opt/rocm/lib -lhsa-amd-aqlprofile64

app8_aql_stop_packet: app8_aql_stop_packet.cpp aql_profiler.cpp aql_profiler.h
	$(CC) $(CFLAGS) -I../../projects/aqlprofile/src/core/include -o $@ app8_aql_stop_packet.cpp aql_profiler.cpp $(LDFLAGS) -L/opt/rocm/lib -lhsa-amd-aqlprofile64

# Clean target
clean:
	rm -f $(APPS)

# Phony targets
.PHONY: all clean