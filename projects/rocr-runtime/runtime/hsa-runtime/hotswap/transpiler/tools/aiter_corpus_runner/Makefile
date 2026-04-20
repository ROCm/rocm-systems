# Build the AITER-corpus-runner-local LD_PRELOAD shim that spoofs
# hipGetDeviceProperties().gcnArchName.  See arch_spoof.cpp for the
# rationale.
#
# Self-contained: no parent CMake, no system installs, no
# dependencies on a Salmon-enabled ROCR build (the shim is a HIP-API
# pass-through; it only needs the public HIP headers from /opt/rocm).
#
# Usage:
#   make                 # builds libaiter_arch_spoof.so
#   make clean
#
# Override ROCM if your HIP headers are elsewhere:
#   make ROCM=/opt/rocm-7.2.1

CXX   ?= g++
ROCM  ?= /opt/rocm
WARN  := -Wall -Wextra -Wno-unused-parameter
OPT   := -O2
# -g1 keeps line-number info in the resulting .so (useful for
# perf / gdb triage without bloating the file), without the full
# type/scope tables of -g3.  The LD_PRELOAD shim is tiny, so the
# cost is a few dozen kB.
DBG   := -g1
STD   := -std=c++17
# -fvisibility=hidden hides every symbol by default; the explicit
# ``__attribute__((visibility("default")))`` tags on our HIP hooks
# are then the only things the dynamic linker sees from this .so.
# That pairs with arch_spoof.ver — the version script does the
# version-matching, the visibility flag ensures nothing else leaks
# out (incidental std::* template instantiations, our self-check
# helpers, etc.).
VIS   := -fvisibility=hidden -fvisibility-inlines-hidden

.PHONY: all clean
all: libaiter_arch_spoof.so

# --version-script is mandatory: without it our exports land as
# plain-named symbols that the dynamic linker won't match against
# AITER's ``hipGetDevicePropertiesR0600@hip_6.0`` versioned undefined
# reference (observed directly: unversioned shim loaded cleanly, shim's
# "active" line printed, but every downstream call resolved to the real
# libamdhip64 and the gfx950 .co never got picked up).  See
# arch_spoof.ver for the version DAG.
#
# We link against libamdhip64 directly (``-L$(ROCM)/lib -lamdhip64``) so
# our undefined references to the *public* HIP entry points
# (hipModuleLoadData, hipModuleGetFunction, hipModuleUnload,
# hipGetErrorString) are satisfied by a DT_NEEDED entry.  That makes
# libamdhip64 a load-time dependency of libaiter_arch_spoof.so — so
# even if someone puts us in LD_PRELOAD without any other HIP consumer,
# the hooks we call into have a known provenance rather than relying on
# the (later-loaded) AITER module_*.so pulling HIP in for us.  Our
# hooks for hipGetDevicePropertiesR0600 etc. still LD_PRELOAD-override
# the real ones, because the dynamic linker walks LD_PRELOAD ahead of
# DT_NEEDED when resolving versioned undefined references in the
# actual caller (AITER's modules).
libaiter_arch_spoof.so: arch_spoof.cpp arch_spoof.ver
	$(CXX) -shared -fPIC $(STD) $(OPT) $(DBG) $(VIS) $(WARN) \
	  -D__HIP_PLATFORM_AMD__ \
	  -I$(ROCM)/include \
	  -L$(ROCM)/lib \
	  -Wl,--version-script=arch_spoof.ver \
	  -o $@ $< \
	  -ldl -lamdhip64

clean:
	rm -f libaiter_arch_spoof.so
