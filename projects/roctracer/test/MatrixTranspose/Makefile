ROOT_PATH = ../..
LIB_PATH  = $(ROOT_PATH)/build
LIB_NAME  = roctracer64
ROC_LIBS  = -L$(LIB_PATH) -l$(LIB_NAME)

EXECUTABLE = ./MatrixTranspose
SOURCES    = MatrixTranspose.cpp
OBJECTS    = $(SOURCES:.cpp=.o)

ITERATIONS ?= 100

HIP_PATH ?= /opt/rocm/hip
HIPCC     = $(HIP_PATH)/bin/hipcc

CXX=$(HIPCC)
CXXFLAGS =-g -I$(ROOT_PATH) -I$(ROOT_PATH)/inc -DITERATIONS=$(ITERATIONS)

export LD_LIBRARY_PATH=$(LIB_PATH)

all: clean $(EXECUTABLE) test

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(HCC_LIBS) $(ROC_LIBS)

test: $(EXECUTABLE)
	HCC_PROFILE=1 $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)

.PHONY: all test clean
