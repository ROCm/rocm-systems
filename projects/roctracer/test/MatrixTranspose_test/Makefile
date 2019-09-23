ROOT_PATH = ../..
LIB_PATH  = $(ROOT_PATH)/build
ROC_LIBS  = -L$(LIB_PATH) -lroctracer64 -lroctx64

ITERATIONS ?= 100

HCC_HOME  ?= /opt/rocm/hcc
HIP_PATH  ?= /opt/rocm/hip

HIPCC=$(HIP_PATH)/bin/hipcc
CXX       = $(HIPCC)
CXXFLAGS  = -g -I$(ROOT_PATH) -I$(ROOT_PATH)/inc -DLOCAL_BUILD=1 -DITERATIONS=$(ITERATIONS)

TARGET=hcc

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixTranspose

export LD_LIBRARY_PATH=$(LIB_PATH)

.PHONY: test


all: $(EXECUTABLE) test


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

