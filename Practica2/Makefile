LINK = nvcc

OUT_FILE = videoReductor

FLAGS = -Xcompiler -fopenmp -std=c++11
OPENCV = `pkg-config --cflags --libs opencv4`

all: videoReductor

videoReductor: videoReductor.cu
	$(LINK) -o $(OUT_FILE) $(FLAGS) $(OPENCV) $^

clean: 
	rm -f *.o *~ core