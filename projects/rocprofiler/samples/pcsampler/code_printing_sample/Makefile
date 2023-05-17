# -*- makefile-gmake -*-

ROCM_PATH ?= /opt/rocm
HIP_PATH ?= $(ROCM_PATH)/hip
HIPCC := $(HIP_PATH)/bin/hipcc

ROCM_PATH ?=/opt/rocm
ROCPROFILER_LIBS_PATH ?=$(ROCM_PATH)/lib
ROCPROFILER_INCLUDES=$(ROCPROFILER_LIBS_PATH)/../include/rocprofiler/

ifndef ROCPROFILER_PATH
$(warning You may need to set ROCPROFILER_PATH to the path of the rocprofiler source)
endif

CXXFLAGS += -std=c++17 -Wall

ifdef DEBUG
CXXFLAGS += -gdwarf-4 -O0
else
ifdef DEBUGOPT
CXXFLAGS += -gdwarf-4 -Og
else
CXXFLAGS += -gdwarf-4 -O2
endif
endif

###

srcs := $(wildcard *.cpp)
prog := main

objs := $(srcs:%.cpp=%.o)
deps := $(srcs:%.cpp=%.d)

# Kernel program

CPPFLAGS += -DHAVE_MEMFD_CREATE

$(prog): CC = $(HIPCC)
$(prog): CPPFLAGS += -I$(ROCPROFILER_INCLUDES) -I$(ROCM_PATH)/include
$(prog): LDFLAGS := -L$(ROCPROFILER_LIBS_PATH) -L$(ROCM_PATH)/lib
$(prog): LDLIBS += -ldl -lpthread -lhsa-runtime64 -lrocprofiler64v2 -lrocm-dbgapi -ldw -lelf
$(objs): CXX = $(HIPCC)

# Targets

all: $(prog)

$(prog): $(objs)

-include $(deps)

OUTPUT_OPTION = -MMD -MP -o $@

%.so: %.o
	$(LINK.o) $(OUTPUT_OPTION) $^ $(LDLIBS)

#COMPILE.hip = $(COMPILE.cpp)
#LINK.hip = $(LINK.cpp)

#%.o: %.hip
#	$(COMPILE.hip) $(OUTPUT_OPTION) $<

clean:
	$(RM) $(prog) $(objs) $(deps)

distclean: | clean
	$(RM) compile_commands.json

.PHONY: all clean distclean
