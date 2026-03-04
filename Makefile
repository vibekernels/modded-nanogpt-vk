# Makefile for pure CUDA C GPT training
# Targets H100 (sm_90a) with cuBLAS, cuDNN, cuRAND

NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_90a --use_fast_math \
             -Xcompiler -fPIC -Xcompiler -O3 \
             --expt-relaxed-constexpr \
             -lineinfo

# CUDA toolkit paths (adjust for your system)
CUDA_HOME ?= /usr/local/cuda
CUDNN_HOME ?= $(CUDA_HOME)

INCLUDES = -I$(CUDA_HOME)/include -I$(CUDNN_HOME)/include
LDFLAGS = -L$(CUDA_HOME)/lib64 -L$(CUDNN_HOME)/lib64 \
          -lcudart -lcublas -lcublasLt -lcudnn -lcurand -lm -lpthread

# cudnn-frontend headers (header-only C++ library from NVIDIA/cudnn-frontend)
# Clone: git clone https://github.com/NVIDIA/cudnn-frontend.git
CUDNN_FRONTEND_DIR ?= cudnn-frontend/include
INCLUDES += -I$(CUDNN_FRONTEND_DIR)

TARGET = train_gpt
SRCS = train_gpt.cu kernels.cu
HEADERS = train_gpt.h kernels.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(SRCS) $(LDFLAGS)

# Separate compilation (useful for faster rebuilds)
kernels.o: kernels.cu kernels.h train_gpt.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ kernels.cu

train_gpt.o: train_gpt.cu train_gpt.h kernels.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c -o $@ train_gpt.cu

$(TARGET)_separate: kernels.o train_gpt.o
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) kernels.o train_gpt.o $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

# Debug build
debug: NVCC_FLAGS = -O0 -std=c++17 -arch=sm_90a -g -G -lineinfo
debug: $(TARGET)

# Profile build
profile: NVCC_FLAGS += -DPROFILE
profile: $(TARGET)

# Print GPU info
info:
	nvidia-smi
	$(NVCC) --version
