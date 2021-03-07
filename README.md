# Simple-Sparsely-Connected-NN
Simple C++ implementation of a sparsely connected multi-layer neural network using OpenMP and CUDA for parallelization.

## OpenMP
This version of the program uses OpenMP to achieve parallelism.

### Compile and Run
Steps:
```
cd OpenMP/src
g++ -fopenmp main.cpp NeuralNet.cpp Vector.cpp -o openmp 
OMP_NUM_THREADS=p ./openmp [N] [K] [p_mode] [v_mode]
```
### Arguments
- N, Specify the number of nodes (neurons) in the input layer, it should be a positive integer
- K, Specify the number of layers of the NN, it should be an integer greater than 1
- p_mode (optional)\tSpecify the kind of parallelization technique, accepted values: {0, 1, 2},
	if 0 (default) it executes the sequential version,
	if 1 it parallelizes the outer for loop,
	if 2 it parallelizes the inner for loop and applies a reduction
- v_mode (optional)\tSpecify the kind of interaction with the user, accepted values: {0, 1}
	if 0 (default) it prints only the execution time,
	if 1 it prints additional information in addition to the execution time

The scripts `OpenMP/straong_scaling.sh` and `OpenMP/weak_scaling.sh` can be used to automate the execution of the program varying the number of threads and the problem size. They produce recording files which then could be used to compute speedup and strong/weak scaling efficiency.

## CUDA
This version of the program uses CUDA to achieve parallelism.

### Compile and Run
Steps:
```
cd OpenMP/src
nvcc -std=c++11 main.cu NeuralNet.cu Vector.cu -o cuda
./cuda [N] [K] [s_mode] [v_mode]
```
### Arguments
- N, Specify the number of nodes (neurons) in the input layer, it should be a positive integer
- K, Specify the number of layers of the NN, it should be an integer greater than 1
- s_mode (optional)\tSpecify if use the shared memory or not, accepted values: {0, 1},
	if 0 the available shared memory is NOT exploited,
	if 1 (default) the available shared memory is exploited,
- v_mode (optional)\tSpecify the kind of interaction with the user, accepted values: {0, 1}
	if 0 (default) it prints only the execution time,
	if 1 it prints additional information in addition to the execution time

The script `CUDA/eval.sh` automates the execution of the program varying the problem size. It produces a recording file which then could be used to compute speedup w.r.t the OpenMP version.

## Report
The file `Report.pdf` contains an in-depth analysis of the parallel algorithms.
