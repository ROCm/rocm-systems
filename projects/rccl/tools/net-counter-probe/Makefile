CXX ?= g++
CXXFLAGS ?= -std=c++14 -O2 -Wall -Wextra

net-counter-probe: net_counter_probe.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f net-counter-probe

.PHONY: clean
