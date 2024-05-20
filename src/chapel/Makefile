#DEBUG=true
CHPL_FLAGS+=--detailed-errors --savec tmp --print-commands --fast
CHPL_SOURCE=main.chpl
ifeq ($(CHPL_LOCALE_MODEL),gpu)
# GPU version
ifndef CHPL_GPU
$(error CHPL_GPU not defined)
else
CHPL_FLAGS+=--report-gpu -suseGPU=true --ldflags -no-pie
ifeq ($(CHPL_GPU),nvidia)
LDFLAGS+=-L$(CUDA_PATH)/lib64 -lcudart
else
ifeq ($(CHPL_GPU),amd)
LDFLAGS+=-lamdhip64
else
$(error Unknown value for CHPL_GPU: $(CHPL_GPU))
endif
endif
ifndef CHPL_GPU_ARCH
$(error CHPL_GPU_ARCH not defined)
endif
endif
else
# CPU version
ifdef CHPL_GPU
$(error CHPL_GPU is set, but CHPL_LOCALE_MODEL!= gpu)
endif
endif

EXE = chapel-reduced

.PHONY: all $(BENCHMARK_EXE) clean

all: $(EXE)

run:	$(EXE)
	./$(EXE) dot 268435456

$(EXE):	$(CHPL_SOURCE)
	chpl $(CHPL_FLAGS) $(CHPL_SOURCE) $(LDFLAGS) -o $@

.PHONY:	clean
clean:	
	rm -rf $(EXE)