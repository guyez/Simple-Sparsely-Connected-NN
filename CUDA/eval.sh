#!/usr/bin/env bash

N=5000
K=250

echo "1D Stencil Kernel"

for i in {1..7}; do
	echo "N = $N, K = $K"
	./src/cuda "$N" "$K" 0
	N=$((N * 2))
done

echo "1D Stencil Kernel with shared memory"

for i in {1..7}; do
        echo "N = $N, K = $K"
        ./src/cuda "$N" "$K" 1
        N=$((N * 2))
done

