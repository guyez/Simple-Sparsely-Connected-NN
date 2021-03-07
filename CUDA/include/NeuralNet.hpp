/****************************************************************************
 *
 * NeuralNet.hpp - AP4AI CUDA project
 *
 * Last updated in 2021 by Simone Gayed Said <simone.gayed(at)studio.unibo.it>
 *
 ****************************************************************************/

#ifndef CUDA_NEURALNET_HPP_
#define CUDA_NEURALNET_HPP_

#include "Vector.hpp"
#include <random>

#define R 5

#define BLKDIM 1024

class NeuralNet
{
public:
    // Default constructor to initialize a new neural net object
    NeuralNet(std::vector<unsigned int> topology, std::uniform_real_distribution<float> distr, std::mt19937 &generator);

    // Feed forward the input Vector and return the net's prediction
    Vector queryNet(const Vector &inputList, unsigned int s_mode);

    // Move a new neural net object from the Device(CUDA GPU) to the Host
    void to_host();

    // Move a new neural net object from the Host to the Device(CUDA GPU)
    void to_cuda();                                               
     

private:
    // Function used to randomly initialize the weight and bias vectors
    Vector initializeRandom(unsigned int n, std::uniform_real_distribution<float> distr, std::mt19937 &generator) const;

    Vector bias_;
    std::vector<unsigned int> topology_;
    std::vector<Vector> weights_;
    std::vector<Vector> outputs_;

    bool on_cuda_;    
          
};

#endif /* CUDA_NEURALNET_HPP_ */
