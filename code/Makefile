CXX := g++
CXXFLAGS := -std=c++17 -O3 
LOCALFLAGS := -I/opt/homebrew/Cellar/libpng/1.6.44/include -L/opt/homebrew/Cellar/libpng/1.6.44/lib
NVFLAGS := -std=c++11 -O3 -Xptxas=-v -arch=sm_61
LDFLAGS := -lpng -lz
HIPFLAGS := $(CXXFLAGS)
SEQ_TARGET := Gaussian_seq
CUDA_TARGET := Gaussian_cuda
HIP_TARGET := Gaussian_hip

seq: Gaussian_seq.cpp
	$(CXX) $(CXXFLAGS) $(LOCALFLAGS) $(LDFLAGS) -o $(TARGET) Gaussian_seq.cpp

amd: Gaussian.cpp
	hipcc $(HIPFLAGS) $(LDFLAGS) -o $(SEQUENTIAL) Gaussian.cpp

cuda: Gaussian.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $(TARGET) Gaussian.cu

clean:
	rm -f $(SEQ_TARGET) $(CUDA_TARGET) $(HIP_TARGET)
