ROOT_PATH=../..
LIB_PATH=$(ROOT_PATH)/b
LIB_NAME=roctracer64

export HCC_HOME=/home/evgeny/git/compute/out/ubuntu-16.04/16.04/hcc
HIP_PATH=/home/evgeny/git/compute/external/hip/hip
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixTranspose
export LD_LIBRARY_PATH=$(LIB_PATH)

.PHONY: test


all: $(EXECUTABLE) test

CXXFLAGS =-g -DCOMPILE_HIP_ATP_MARKER=1 -I$(ROOT_PATH)
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ -L/home/evgeny/git/compute/out/ubuntu-16.04/16.04/hcc/lib -lmcwamp_hsa -L$(LIB_PATH) -l$(LIB_NAME)


test: $(EXECUTABLE)
	HCC_PROFILE=1 $(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

