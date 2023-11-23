LINK = nvcc

OUT_FILE = piCuda 

FLAGS = -Xcompiler -fopenmp -std=c++11
OPENCV = `pkg-config --cflags`

all: piCuda

piCuda: piCuda.cu
	$(LINK) -o $(OUT_FILE) $(FLAGS) $(OPENCV) $^

clean: 
	rm -f *.o *~ core