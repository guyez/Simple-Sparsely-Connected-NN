/****************************************************************************
 *
 * main.cu - AP4AI CUDA project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 * nvcc -std=c++11 main.cu NeuralNet.cu Vector.cu -o cuda
 *
 * Run with:
 * ./cuda [N] [K] [s_mode] [v_mode]
 *
 ****************************************************************************/

#include <climits>
#include <iostream>
#include <string>
#include "../include/NeuralNet.hpp"

static void show_usage()
{
    std::cerr << "Usage: "
              << "./main N K s_mode v_mode\n"
              << "Arguments:\n"
              << "\tN\tSpecify the number of nodes (neurons) in the input layer, it should be a positive integer\n"
              << "\tK\tSpecify the number of layers of the NN, it should be an integer greater than 1\n"
              << "\ts_mode (optional)\tSpecify if use the shared memory or not, accepted values: {0, 1},\n"
              << "\t\t if 0 the available shared memory is NOT exploited,\n"
              << "\t\t if 1 (default) the available shared memory is exploited,\n"
              << "\tv_mode (optional)\tSpecify the kind of interaction with the user, accepted values: {0, 1},\n"
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

    unsigned int s_mode;
    if (argc == 4)
    {
        s_mode = strtoul(argv[3], &p, 10);
        if (s_mode != 0 && s_mode != 1 || *p)
        {
            show_usage();
            return 0;
        }
    }
    else
        s_mode = 1;

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

    std::mt19937 generator(42); //set the seed
    std::uniform_real_distribution<float> distr(-1, 1);

    NeuralNet *NNs = new NeuralNet(topology, distr, generator);

    NNs->to_cuda();

    Vector testInput(N);
    for (int n = 0; n < testInput.getNumOfElements(); n++)
    {
        testInput[n] = distr(generator);
    }

    double tstart, tend;

    tstart = hpc_gettime();

    Vector result = NNs->queryNet(testInput, s_mode);

    tend = hpc_gettime();

    printf("Execution time: %f seconds\n", tend - tstart);

    if (v_mode)
    {
        std::cout << "Input: " << std::endl;
        testInput.printVector();
        
        std::cout << "Output: " << std::endl;
        result.printVector();
   
	    // Bring the neural network back to host to test the CPU version
        NNs->to_host();

        Vector result_check = NNs->queryNet(testInput, s_mode);

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
