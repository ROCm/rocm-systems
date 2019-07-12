HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

# specify .cpp filename here
filename=cache

SOURCES=$(filename).cpp

all: $(filename)

$(filename): $(SOURCES)
	$(HIPCC) $(CXXFLAGS) $(SOURCES) -o $@

clean:
	rm -f *.o *.out $(filename)
