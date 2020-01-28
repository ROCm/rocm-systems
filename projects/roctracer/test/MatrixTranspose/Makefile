ROOT_PATH = ../..
LIB_PATH  = $(ROOT_PATH)/build
ROC_LIBS  = -Wl,--rpath,${LIB_PATH} $(LIB_PATH)/libroctracer64.so $(LIB_PATH)/libroctx64.so

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

CXXFLAGS =-g -I$(ROOT_PATH) -I$(ROOT_PATH)/inc -DLOCAL_BUILD=1
CXX=$(HIPCC)

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(ROC_LIBS)

test: $(EXECUTABLE)
	$(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

