#!/bin/sh

CORES=`cat /proc/cpuinfo | grep processor | wc -l` # number of cores

echo "--- Weak scaling efficiency test (N) --- \n"

PROB_SIZE_N_0=2000 
PROB_SIZE_K=500

for p in `seq $CORES`; do
	echo -n "\nUsing $p OpenMP thread(s):\n\n"
	PROB_SIZE_N_P=$(( (1 - p) * PROB_SIZE_K + p * PROB_SIZE_N_0 ))
	echo -n "Input layer nodes: ${PROB_SIZE_N_P}\nNumber of layers: ${PROB_SIZE_K}\n\n"
	for _ in `seq 10`; do
		EXEC_TIME_1="$( OMP_NUM_THREADS=$p ./src/openmp $PROB_SIZE_N_P $PROB_SIZE_K 1 | sed 's/Execution time //' )"
		EXEC_TIME_2="$( OMP_NUM_THREADS=$p ./src/openmp $PROB_SIZE_N_P $PROB_SIZE_K 2 | sed 's/Execution time //' )"
		
		echo -n "${EXEC_TIME_1}\t${EXEC_TIME_2}\n"
	done
done















