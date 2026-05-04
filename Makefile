NVCC = nvcc
CXX = g++
NVCC_FLAGS = -arch=sm_75 # Adjust this based on your GPU's compute capability (e.g., sm_75 for Turing, sm_86 for Ampere)
CXX_FLAGS = -O3 -Wall
OPENMP_FLAGS = -fopenmp
LDFLAGS = -lcudart

TARGET = benchmark
RESULTS_FILE = benchmark_results.csv
PLOT_SCRIPT = plot_speedup.py

all: $(TARGET)

$(TARGET): benchmark.cu
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) benchmark.cu -Xcompiler "$(OPENMP_FLAGS)" $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) > $(RESULTS_FILE)

plot: $(RESULTS_FILE) $(PLOT_SCRIPT)
	python3 $(PLOT_SCRIPT) $(RESULTS_FILE)

clean:
	rm -f $(TARGET) $(RESULTS_FILE)