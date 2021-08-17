CXXFLAGS += -std=c++17 -O3 -Wno-unused-result -g
LINK_FLAGS :=

all: transpose

debug: CXXFLAGS += -ggdb -g
debug: transpose

transpose: transpose.cpp
	hipcc ${CXXFLAGS} -o $@ $^ ${LINK_FLAGS}

.PHONY: clean

run:
	./transpose

run.verbose:
	AMD_LOG_LEVEL=3, AMD_LOG_MASK=128 ./transpose

rd:
	rocgdb --args ./transpose

prof.trace:
	rocprof --hip-trace --stats ./transpose

prof:
	rocprof --obj-tracking on -i profile.txt ./transpose

prof.full:
	rocprof --obj-tracking on -i profile.txt  -m custom_metric.xml ./transpose

callgrind:
	valgrind --tool=callgrind ./transpose

callgrind.ann:
	callgrind_annotate --inclusive=yes --tree=both callgrind.out.* > callgrind.out

clean:
	rm -f transpose *.o results.*

