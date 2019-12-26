ROOT_PATH = ../..
LIB_PATH  = $(ROOT_PATH)/build
ROC_LIBS  = -Wl,--rpath,${LIB_PATH} $(LIB_PATH)/libroctracer64.so $(LIB_PATH)/libroctx64.so
HSA_KMT_INC_PATH ?= /opt/rocm/include
HIP_VDI ?= 0
ITERATIONS ?= 100

HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc



FLAGS =-g -I$(ROOT_PATH) -I$(ROOT_PATH)/inc -I${HSA_KMT_INC_PATH} -DLOCAL_BUILD=1 -DHIP_VDI=${HIP_VDI} -DITERATIONS=$(ITERATIONS)
ifeq ($(C_TEST), 1)
	COMP=gcc
	SOURCES = MatrixTranspose.c
	FLAGS += -DHIP_TEST=0 -D__HIP_PLATFORM_HCC__=1 -I/opt/rocm/hcc/include
	EXECUTABLE=./MatrixTranspose_ctest
else
	COMP=$(HIPCC)
	FLAGS += -DHIP_TEST=1
	SOURCES = MatrixTranspose.cpp
	EXECUTABLE=./MatrixTranspose
endif
OBJECTS = MatrixTranspose.o

.PHONY: test

all: clean $(EXECUTABLE)

$(OBJECTS): $(SOURCES)
	$(COMP) $(FLAGS) -c -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test: $(EXECUTABLE)
	LD_PRELOAD=$(LIB_PATH)/libkfdwrapper64.so $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

