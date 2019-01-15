HCC_HOME? = /opt/rocm/hcc
HIP_PATH? = /opt/rocm/hip

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE=./MatrixTranspose

HIPCC=$(HIP_PATH)/bin/hipcc
CXXFLAGS =-g
CXX=$(HIPCC)

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@


test: $(EXECUTABLE)
	LD_PRELOAD=$(HCC_HOME)/lib/libmcwamp_hsa.so $(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

.PHONY: all test clean
