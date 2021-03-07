/****************************************************************************
 *
 * main.cpp - AP4AI OpenMP project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * g++ -fopenmp main.cpp NeuralNet.cpp Vector.cpp -o openmp
 *
 * Run with:
 * OMP_NUM_THREADS=p ./openmp [N] [K] [p_mode] [v_mode]
 *
 ****************************************************************************/

#include <climits>
#include <iostream>
#include <string>
#include "../include/NeuralNet.hpp"

static void show_usage()
{
    std::cerr << "Usage: "
              << "OMP_NUM_THREADS=p ./main N K p_mode v_mode\n"
              << "Arguments:\n"
              << "\tN\tSpecify the number of nodes (neurons) in the input layer, it should be a positive integer\n"
              << "\tK\tSpecify the number of layers of the NN, it should be an integer greater than 1\n"
              << "\tp_mode (optional)\tSpecify the kind of parallelization technique, accepted values: {0, 1, 2},\n"
              << "\t\t if 0 (default) it executes the sequential version,\n"
              << "\t\t if 1 it parallelizes the outer for loop,\n"
              << "\t\t if 2 it parallelizes the inner for loop and applies a reduction\n"
              << "\tv_mode (optional)\tSpecify the kind of interaction with the user, accepted values: {0, 1}\n"
              << "\t\t if 0 (default) it prints only the execution time,\n"
              << "\t\t if 1 it prints additional information in addition to the execution time\n"
              << std::endl;
}

std::vector<unsigned int> get_topology(unsigned int N, unsigned int K)
{
    if (N <= (K - 1) * (R - 1))
    {
        std::cerr << "The of input neurons is too low for the defined shrinking factor and number of given layers" << std::endl;
        show_usage();
        exit(2);
    }

    std::vector<unsigned int> topology;

    unsigned int drop_out = 0;

    for (int i = 0; i < K; i++)
    {
        topology.push_back(N - drop_out);
        drop_out += (R - 1);
    }

    return topology;
}

int main(int argc, const char *argv[])
{

    if (argc < 3 || argc > 5)
    {
        show_usage();
        return 0;
    }

    char* p;

    int N = strtoul(argv[1], &p, 10);
    if (N < R || *p)
    {
        show_usage();
        return 0;
    }
    int K = strtoul(argv[2], &p, 10);
    if (K <= 1 || *p)
    {
        show_usage();
        return 0;
    }

    unsigned int p_mode;
    if (argc == 4)
    {
        p_mode = strtoul(argv[3], &p, 10);
        if (p_mode != 0 && p_mode != 1 && p_mode != 2 || *p)
        {
            show_usage();
            return 0;
        }
    }
    else
        p_mode = 0;

    unsigned int v_mode;
    if (argc == 5)
    {
        v_mode = strtoul(argv[4], &p, 10);
        if (v_mode != 0 && v_mode != 1 || *p)
        {
            show_usage();
            return 0;
        }
    }
    else
        v_mode = 0;

    std::vector<unsigned int> topology = get_topology(N, K);

    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distr(-1, 1);

    NeuralNet *NNs = new NeuralNet(topology, distr, generator);

    Vector testInput(N);
    for (int n = 0; n < testInput.getNumOfElements(); n++)
    {
        testInput[n] = distr(generator);
    }

    double tstart, tend;

    tstart = omp_get_wtime();

    Vector result = NNs->queryNet(testInput, p_mode);

    tend = omp_get_wtime();

    printf("Execution time: %f seconds\n", tend - tstart);

    if (v_mode)
    {
    
        std::cout << "\nInput: " << std::endl;
        testInput.printVector();

        std::cout << "Output: " << std::endl;
        result.printVector();

        Vector result_check = NNs->queryNet(testInput, -1); 
        std::cout << "Check the correctness of the result:" << std::endl;
        if (result == result_check)
        {
            std::cout << "The result is correct!" << std::endl;
        }
        else
            std::cout << "Wrong result!" << std::endl;
        
    }

    return 0;
}
