ROOT_PATH = ../..
LIB_PATH  = $(ROOT_PATH)/build
ROC_LIBS  = -L$(LIB_PATH) -lroctracer64 -lroctx64
HSA_KMT_INC_PATH = /opt/rocm/include
export LD_LIBRARY_PATH=$(LIB_PATH)
ITERATIONS ?= 100

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixTranspose

.PHONY: test


all: clean $(EXECUTABLE)

CXXFLAGS =-g -I$(ROOT_PATH) -I$(ROOT_PATH)/inc -I${HSA_KMT_INC_PATH} -DLOCAL_BUILD=1 -DITERATIONS=$(ITERATIONS)
CXX=$(HIPCC)

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test: $(EXECUTABLE)
	$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

