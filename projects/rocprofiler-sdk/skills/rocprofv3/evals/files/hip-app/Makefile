HIPCC ?= /opt/rocm/bin/hipcc
ARCH  ?= native

app: app.hip
	$(HIPCC) -g -O3 --offload-arch=$(ARCH) -o $@ $< -lrocprofiler-sdk-roctx

clean:
	rm -f app

.PHONY: clean
