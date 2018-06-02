ROOT_PATH=../..
LIB_PATH=$(ROOT_PATH)/b
LIB_NAME=roctracer64
ROC_LIBS=-L$(LIB_PATH) -l$(LIB_NAME)

HIP_PATH?=/home/evgeny/work/roc-1.8.x/hip
HCC_PATH?=/home/evgeny/work/roc-1.8.x/hcc/b

HIPCC=$(HIP_PATH)/bin/hipcc
HCC_LIBS=-L$(HCC_PATH)/lib -lmcwamp_hsa

SOURCES = MatrixTranspose.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./MatrixTranspose
export LD_LIBRARY_PATH=$(LIB_PATH):$(HIP_PATH)/lib:$(HCC_PATH)/lib
export HCC_HOME=$(HCC_PATH)

.PHONY: test

ITERATIONS?=10

all: clean $(EXECUTABLE) test

CXXFLAGS =-g -I$(ROOT_PATH) -I$(ROOT_PATH)/inc -DITERATIONS=$(ITERATIONS)
CXX=$(HIPCC)


$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@ $(HCC_LIBS) $(ROC_LIBS)


test: $(EXECUTABLE)
	HCC_PROFILE=1 $(EXECUTABLE)


clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)
	rm -f $(HIP_PATH)/src/*.o

