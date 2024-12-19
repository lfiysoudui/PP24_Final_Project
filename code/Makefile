CXX := g++
CXXFLAGS := -std=c++17 -O3 
LOCALFLAGS := -I/opt/homebrew/Cellar/libpng/1.6.44/include -L/opt/homebrew/Cellar/libpng/1.6.44/lib
AMD_SERVER_FLAGS := -I/home1/ntustudent001/libpng/include -L/home1/ntustudent001/libpng/lib
NVFLAGS := -std=c++11 -O3 -Xptxas=-v -arch=sm_61
LDFLAGS := -lpng -lz
HIPFLAGS := $(CXXFLAGS)
SEQ_TARGET := Gaussian_seq
CUDA_TARGET := Gaussian_cuda
HIP_TARGET := Gaussian_hip

seq: Gaussian_seq.cpp
	$(CXX) $(CXXFLAGS) $(LOCALFLAGS) $(LDFLAGS) -o $(SEQ_TARGET) Gaussian_seq.cpp

hip: Gaussian_hip.cpp
	hipcc $(HIPFLAGS) $(AMD_SERVER_FLAGS) $(LDFLAGS) -o $(HIP_TARGET) Gaussian_hip.cpp

cuda: Gaussian.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $(CUDA_TARGET) Gaussian.cu

clean:
	rm -f $(SEQ_TARGET) $(CUDA_TARGET) $(HIP_TARGET)
