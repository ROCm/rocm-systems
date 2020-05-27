ROOT_PATH ?= ../..
LIB_PATH  ?= $(ROOT_PATH)/build
ROCM_PATH ?= /opt/rocm
HIP_VDI ?= 0
ITERATIONS ?= 100

ROC_LIBS  = -Wl,--rpath,${LIB_PATH} $(LIB_PATH)/libroctracer64.so $(LIB_PATH)/libroctx64.so

HIP_PATH ?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc
TARGET=hcc

EXECUTABLE=./MatrixTranspose
OBJECTS = MatrixTranspose.o
FLAGS =-g -I$(ROOT_PATH) -I$(ROOT_PATH)/inc -I$(ROCM_PATH)/hsa/include/hsa -I$(ROCM_PATH)/hsa/include -I$(ROCM_PATH)/hip/include -I$(ROCM_PATH)/include -DLOCAL_BUILD=1 -DHIP_VDI=${HIP_VDI} -DITERATIONS=$(ITERATIONS) -DAMD_INTERNAL_BUILD=1

ifeq ($(C_TEST), 1)
	COMP=${CC}
	SOURCES = MatrixTranspose.c
	FLAGS += -DHIP_TEST=0 -D__HIP_PLATFORM_HCC__=1
else
	COMP=$(HIPCC)
	SOURCES = MatrixTranspose.cpp
	FLAGS += -DHIP_TEST=1
endif
ifeq ($(MGPU_TEST), 1)
	FLAGS += -DMGPU_TEST=1
endif
ifeq ($(HIP_API_ACTIVITY_ON), 1)
        FLAGS += -DHIP_API_ACTIVITY_ON=1
endif

.PHONY: test

all: clean $(EXECUTABLE)

$(OBJECTS): $(SOURCES)
	$(COMP) $(FLAGS) -c -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test: $(EXECUTABLE)
	LD_PRELOAD="$(LIB_PATH)/libkfdwrapper64.so librocprofiler64.so" $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

